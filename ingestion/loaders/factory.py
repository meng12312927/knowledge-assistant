"""
统一文档加载器工厂。

核心思想：无论用户上传 PDF、Word 还是 PPT，调用方只关心一个接口：
    chunks = load_document(file_path)

返回的永远是 List[DocumentChunk]，上层模块无需关心原始格式。
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

# DocumentChunk 是项目内部的标准文档单元。
from langchain_core.documents import Document as LangChainDocument

# models/document.py 中定义的 Pydantic 模型
from models.document import DocumentChunk, DocumentType, SourceDocument


class BaseDocumentLoader(ABC):
    """
    所有文档加载器的抽象基类。

    如果你以后要支持新的格式（比如 EPUB），只需要：
    1. 继承 BaseDocumentLoader
    2. 实现 load() 方法
    3. 在 DocumentLoaderFactory 中注册
    """

    @abstractmethod
    def load(self, file_path: str) -> List[DocumentChunk]:
        """
        加载文档并返回标准化的文本块列表。

        Args:
            file_path: 文件的绝对路径或相对路径

        Returns:
            List[DocumentChunk]: 每个 chunk 包含内容和元数据
        """
        pass

    def _make_chunks_from_pages(
        self,
        pages: List[LangChainDocument],
        filename: str,
        doc_type: DocumentType
    ) -> List[DocumentChunk]:
        """
        通用辅助方法：将 LangChain Document 列表转换为我们定义的 DocumentChunk。

        LangChain 的 Document 结构：
        - page_content: str  (文本内容)
        - metadata: dict    (页码、来源等)
        """
        chunks = []
        for i, page in enumerate(pages):
            # 构建元数据：保留原始信息，同时统一添加我们的字段
            metadata = {
                **page.metadata,           # 原始元数据（如页码）
                "source_file": filename,   # 来源文件名
                "doc_type": doc_type.value,# 文档类型
                "page_index": i,           # 在文档中的序号
            }
            chunks.append(DocumentChunk(
                content=page.page_content,
                metadata=metadata
            ))
        return chunks


class PDFLoader(BaseDocumentLoader):
    """
    PDF 文档加载器

    技术选型：PyMuPDF (fitz)
    优势：速度快，能提取页面坐标信息（用于后续判断是否为双栏布局）
    劣势：对扫描版 PDF 需要 OCR（可后续接入 pytesseract）

    表格检测：使用 get_text("dict") 获取带坐标的结构化数据，
    当检测到多行 span 的 X 坐标对齐时，自动转为 Markdown 表格。
    """

    # 表格检测容差参数
    _TABLE_COL_TOLERANCE: float = 15.0  # 同列的 x0 浮动范围（点）
    _TABLE_MIN_ROWS: int = 2            # 最少行数
    _TABLE_MIN_COLS: int = 2            # 最少列数

    def load(self, file_path: str) -> List[DocumentChunk]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF 未安装。请运行：pip install pymupdf"
            )

        doc = fitz.open(file_path)
        filename = Path(file_path).name
        pages = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = self._extract_page_text(page)

            pages.append(LangChainDocument(
                page_content=text,
                metadata={
                    "page_number": page_num + 1,
                    "total_pages": len(doc),
                }
            ))

        doc.close()
        return self._make_chunks_from_pages(
            pages, filename, DocumentType.PDF
        )

    # ═══════════════════════════════════════════════════════════
    # 页面文本提取（含表格检测）
    # ═══════════════════════════════════════════════════════════

    _Y_ROW_TOLERANCE: float = 5.0  # 同行的 Y 坐标浮动范围（点）

    def _extract_page_text(self, page) -> str:
        """提取一页文本：先收集所有 span，分组为「行」，再检测表格。"""
        all_rows = self._collect_all_rows(page)

        # 分组：连续的表格行 vs 普通行
        groups = self._group_table_rows(all_rows)

        result_parts = []
        for group in groups:
            if group["is_table"]:
                md_table = self._build_markdown_table(group["rows"])
                if md_table:
                    result_parts.append(md_table)
            else:
                # 普通文本：每行直接拼接
                text = "\n".join(
                    "".join(span["text"] for span in row["spans"])
                    for row in group["rows"]
                )
                if text.strip():
                    result_parts.append(text)

        text = "\n\n".join(result_parts)
        return self._clean_text(text)

    def _collect_all_rows(self, page) -> list:
        """从页面所有文本 block 中收集 span，按 Y 坐标合并为行。

        同一个 Y 坐标的 span（差值 < _Y_ROW_TOLERANCE）合并到同一行，
        解决 PDF 中每个单元格是独立 line 的问题。
        """
        # 第 1 步：从所有 block 收集所有 span
        all_spans = []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    bbox = span["bbox"]
                    text = span.get("text", "")
                    if not text.strip():
                        continue
                    all_spans.append({
                        "text": text,
                        "x0": round(bbox[0], 1),
                        "x1": round(bbox[2], 1),
                        "y0": round(bbox[1], 1),
                    })

        if not all_spans:
            return []

        # 第 2 步：按 Y 坐标分组（同行合并）
        all_spans.sort(key=lambda s: (s["y0"], s["x0"]))
        rows = []
        current_row_spans = [all_spans[0]]
        current_y0 = all_spans[0]["y0"]

        for span in all_spans[1:]:
            if abs(span["y0"] - current_y0) <= self._Y_ROW_TOLERANCE:
                current_row_spans.append(span)
            else:
                current_row_spans.sort(key=lambda s: s["x0"])
                rows.append({
                    "spans": current_row_spans,
                    "y0": current_y0,
                })
                current_row_spans = [span]
                current_y0 = span["y0"]

        # 最后一行
        current_row_spans.sort(key=lambda s: s["x0"])
        rows.append({"spans": current_row_spans, "y0": current_y0})

        return rows

    def _group_table_rows(self, rows: list) -> list:
        """将行分组：连续的表格行归为一组，普通行各成一组。

        返回格式：
            [
                {"is_table": True,  "rows": [row1, row2, ...]},
                {"is_table": False, "rows": [row3]},
                ...
            ]
        """
        if not rows:
            return []

        groups = []
        table_buffer = []

        for row in rows:
            n_spans = len(row["spans"])

            if len(table_buffer) >= 1:
                # 检查当前行是否能延续表格
                prev_row_spans = table_buffer[-1]["spans"]
                if self._rows_share_columns(prev_row_spans, row["spans"]):
                    table_buffer.append(row)
                    continue
                else:
                    # 表格中断：判断缓冲区是否为表格
                    groups.append({
                        "is_table": len(table_buffer) >= self._TABLE_MIN_ROWS
                                    and len(table_buffer[0]["spans"]) >= self._TABLE_MIN_COLS,
                        "rows": table_buffer,
                    })
                    table_buffer = [row]
            else:
                table_buffer = [row]

        # 处理最后一个缓冲区
        if table_buffer:
            groups.append({
                "is_table": len(table_buffer) >= self._TABLE_MIN_ROWS
                            and len(table_buffer[0]["spans"]) >= self._TABLE_MIN_COLS,
                "rows": table_buffer,
            })

        return groups

    def _rows_share_columns(self, prev_spans: list, curr_spans: list) -> bool:
        """判断两行是否为同一表格的连续行：span 数相同 + 列 X 坐标对齐。"""
        if len(prev_spans) != len(curr_spans):
            return False
        if len(prev_spans) < self._TABLE_MIN_COLS:
            return False

        for p, c in zip(prev_spans, curr_spans):
            if abs(p["x0"] - c["x0"]) > self._TABLE_COL_TOLERANCE:
                return False
        return True

    @staticmethod
    def _build_markdown_table(rows: list) -> str:
        """将表格行转为 Markdown 表格字符串。"""
        if not rows:
            return ""

        n_cols = len(rows[0]["spans"])

        # 提取每行每列的文本
        cells_by_row = []
        for row in rows:
            cells = []
            for span in row["spans"]:
                cell_text = span["text"].strip().replace("|", "\\|")
                cells.append(cell_text)
            cells_by_row.append(cells)

        # 构建 Markdown 表格
        lines_md = []
        # 第一行作为表头
        lines_md.append("| " + " | ".join(cells_by_row[0]) + " |")
        # 分隔行
        lines_md.append("|" + "|".join([" --- " for _ in range(n_cols)]) + "|")
        # 数据行
        for cells in cells_by_row[1:]:
            while len(cells) < n_cols:
                cells.append("")
            lines_md.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines_md)

    def _clean_text(self, text: str) -> str:
        """基础文本清洗：去除多余空行、统一换行符。保留 Markdown 表格格式。"""
        lines = [line.strip() for line in text.splitlines()]
        # 不去除空白行，保留表格与段落之间的分隔
        lines = [line for line in lines if line]
        return "\n".join(lines)


class MarkdownLoader(BaseDocumentLoader):
    """
    Markdown 加载器

    Markdown 的特殊价值：标题层级（# ## ###）是天然的分块边界信号。
    这里保留标题信息，供后续分块策略使用。
    """

    def load(self, file_path: str) -> List[DocumentChunk]:
        filename = Path(file_path).name

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Markdown 作为一个整体加载，因为分块策略会智能处理标题
        pages = [LangChainDocument(
            page_content=content,
            metadata={
                "file_path": file_path,
                # 预留：后续可用正则提取所有标题，用于增强检索
                "has_headers": content.startswith("#"),
            }
        )]

        return self._make_chunks_from_pages(
            pages, filename, DocumentType.MD
        )


class TXTLoader(BaseDocumentLoader):
    """
    纯文本加载器（不依赖任何外部库）

    直接按行读取文本文件，按段落分割。
    适用于 .txt / .csv / .log 等纯文本格式。
    """

    name = "txt"
    description = "加载纯文本文件"

    def load(self, file_path: str) -> List[DocumentChunk]:
        filename = Path(file_path).name

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 按段落分割（空行分隔）
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        chunks = []
        for i, para in enumerate(paragraphs):
            chunks.append(DocumentChunk(
                content=para,
                metadata={
                    "source_file": filename,
                    "doc_type": "txt",
                    "paragraph_index": i,
                }
            ))

        return chunks


class UnstructuredLoader(BaseDocumentLoader):
    """
    万能加载器（基于 unstructured 库）

    支持：PDF, DOCX, PPTX, XLSX, HTML, TXT, CSV...
    优势：能识别表格、图片标题、列表等丰富元素类型
    劣势：依赖较多，首次安装和运行较慢

    安装命令：
        pip install unstructured[all-docs]
        # Windows 下可能需要额外安装 poppler（PDF）和 tesseract（OCR）
    """

    def load(self, file_path: str) -> List[DocumentChunk]:
        try:
            from unstructured.partition.auto import partition
        except ImportError:
            raise ImportError(
                "unstructured 未安装。请运行：pip install unstructured"
            )

        filename = Path(file_path).name
        suffix = Path(file_path).suffix.lower()

        # partition 是万能入口，根据文件扩展名自动选择解析器
        elements = partition(filename=file_path)

        # unstructured 的 Element 有 .text 和 .category（如 Title, NarrativeText, Table）
        # 我们把相同类型的连续元素合并为一个 chunk
        chunks = []
        current_text = []
        current_category = None

        for element in elements:
            category = str(type(element)).split(".")[-1].strip("'>")
            text = str(element)

            if not text.strip():
                continue

            # 标题单独成块（对检索很有价值）
            if "Title" in category:
                if current_text:
                    chunks.append(self._create_chunk(
                        current_text, current_category, filename
                    ))
                chunks.append(self._create_chunk(
                    [text], category, filename, is_title=True
                ))
                current_text = []
                current_category = None
            else:
                if current_category != category and current_text:
                    chunks.append(self._create_chunk(
                        current_text, current_category, filename
                    ))
                    current_text = []
                current_text.append(text)
                current_category = category

        # 处理最后一批
        if current_text:
            chunks.append(self._create_chunk(
                current_text, current_category, filename
            ))

        return chunks

    def _create_chunk(
        self,
        texts: List[str],
        category: Optional[str],
        filename: str,
        is_title: bool = False
    ) -> DocumentChunk:
        content = "\n".join(texts)
        return DocumentChunk(
            content=content,
            metadata={
                "source_file": filename,
                "element_category": category,
                "is_title": is_title,
            }
        )


# ═══════════════════════════════════════════════════════════
# 工厂类：统一入口
# ═══════════════════════════════════════════════════════════

class DocumentLoaderFactory:
    """
    文档加载器工厂

    使用方式：
        loader = DocumentLoaderFactory.get_loader("doc.pdf")
        chunks = loader.load("doc.pdf")
    """

    # 扩展名 → 加载器类的映射表
    _registry = {
        ".pdf": PDFLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
        # 更多格式默认用 unstructured 处理
        ".docx": UnstructuredLoader,
        ".doc": UnstructuredLoader,
        ".pptx": UnstructuredLoader,
        ".ppt": UnstructuredLoader,
        ".xlsx": UnstructuredLoader,
        ".xls": UnstructuredLoader,
        ".html": UnstructuredLoader,
        ".htm": UnstructuredLoader,
        ".txt": TXTLoader,
        ".csv": UnstructuredLoader,
    }

    @classmethod
    def get_loader(cls, file_path: str) -> BaseDocumentLoader:
        """根据文件扩展名返回对应的加载器实例"""
        suffix = Path(file_path).suffix.lower()

        if suffix not in cls._registry:
            raise ValueError(f"不支持的文件格式: {suffix}。路径: {file_path}")

        loader_class = cls._registry[suffix]
        return loader_class()

    @classmethod
    def register(cls, extension: str, loader_class: type):
        """
        注册新的加载器（扩展用）

        示例：
            DocumentLoaderFactory.register(".epub", EPUBLoader)
        """
        cls._registry[extension.lower()] = loader_class
