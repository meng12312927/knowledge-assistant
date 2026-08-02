"""
Dense 与 BM25 混合检索器。

结合 Dense（向量）检索和 Sparse（BM25/关键词）检索，
解决单一检索方式的盲区。

核心接口：
    retriever = HybridRetriever(vector_store, texts_for_bm25)
    results = retriever.retrieve(query, query_embedding, top_k=10)
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any
import hashlib
import time

from models.document import RetrievedChunk
from rag.chunk_identity import stable_chunk_id
from vectorstore.factory import BaseVectorStore


class BM25Retriever:
    """
    BM25 关键词检索器（Sparse Retrieval）

    BM25 是信息检索领域的经典算法，基于词频和逆文档频率。
    优点：对精确匹配（如产品型号、法律条文编号）非常敏感。
    缺点：不理解语义，"退款"和"退货"被视为不同词。

    安装依赖：pip install rank-bm25
    """

    def __init__(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Args:
            texts: 所有 chunk 的文本内容（用于构建词频统计）
            metadatas: 对应的元数据
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("请安装 rank-bm25: pip install rank-bm25")

        self.texts = texts
        self.metadatas = metadatas or [{} for _ in texts]

        # 简单的中文分词：按字符切分（对 BM25 来说足够用，也可用 jieba 更精准）
        tokenized_corpus = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        """文本分词：中文字符级 + 英文单词级"""
        import re
        # 提取中文字符和英文单词
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+|\d+', text.lower())
        return tokens

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        执行 BM25 检索

        Returns:
            按 BM25 分数排序的 RetrievedChunk，score 做了归一化到 [0, 1]
        """
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 取 Top-K
        top_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]

        results = []
        max_score = max(scores) if max(scores) > 0 else 1.0

        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append(RetrievedChunk(
                content=self.texts[idx],
                metadata=self.metadatas[idx],
                score=float(scores[idx] / max_score)  # 归一化到 [0, 1]
            ))

        return results


class HybridRetriever:
    """
    混合检索器：Dense + Sparse

    召回阶段使用两种检索方式，通过 RRF 合并、去重并排序。
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        bm25_retriever: Optional[BM25Retriever] = None,
        parallel_workers: int = 8,
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self._executor = ThreadPoolExecutor(
            max_workers=max(2, parallel_workers),
            thread_name_prefix="rag-retrieval",
        )

    def retrieve(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10,
        dense_weight: float = 0.7,
        filter_dict: Optional[Dict[str, Any]] = None,
        return_trace: bool = False,
    ):
        """
        执行混合检索

        Args:
            query: 用户原始查询文本（用于 BM25）
            query_embedding: 查询的向量（用于 Dense 检索）
            top_k: 最终返回的结果数（合并去重后）
            dense_weight: Dense 检索结果的权重（0~1），Sparse 权重 = 1 - dense_weight
            filter_dict: 元数据过滤条件

        合并策略：RRF（Reciprocal Rank Fusion）
        RRF 公式：score = Σ 1 / (k + rank)
        优点：不需要统一两种检索的分数尺度，只利用排序位置
        """
        result = self.retrieve_many(
            queries=[query],
            query_embeddings=[query_embedding],
            top_k=top_k,
            filter_dict=filter_dict,
            return_trace=return_trace,
        )[0]
        if return_trace:
            return result
        return result

    def retrieve_many(
        self,
        queries: List[str],
        query_embeddings: List[List[float]],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        return_trace: bool = False,
    ) -> List[Any]:
        """并行执行多个 Query 的 Dense/BM25 通道，并按输入顺序稳定返回。"""
        if len(queries) != len(query_embeddings):
            raise ValueError("Query 数量与 Query Embedding 数量不一致")
        if not queries:
            return []

        # 一个请求内固定 BM25 快照，避免摄取热更新造成不同变体跨版本。
        bm25_snapshot = self.bm25_retriever
        batch_started = time.perf_counter()

        def timed_dense(embedding):
            started = time.perf_counter()
            items = self.vector_store.similarity_search(
                query_embedding=embedding,
                top_k=top_k,
                filter_dict=filter_dict,
            )
            return items, started, time.perf_counter() - started

        def timed_bm25(query):
            started = time.perf_counter()
            items = bm25_snapshot.retrieve(query, top_k=top_k) if bm25_snapshot else []
            if filter_dict:
                items = [item for item in items if self._matches_filter(item, filter_dict)]
            return items, started, time.perf_counter() - started

        futures = []
        for query, embedding in zip(queries, query_embeddings):
            dense_future = self._executor.submit(timed_dense, embedding)
            sparse_future = self._executor.submit(timed_bm25, query)
            futures.append((dense_future, sparse_future))

        outputs = []
        for dense_future, sparse_future in futures:
            dense_results, dense_started, dense_duration = dense_future.result()
            sparse_results, sparse_started, sparse_duration = sparse_future.result()
            fused_results = self._rrf_fusion(dense_results, sparse_results, top_k, k=60)
            if not return_trace:
                outputs.append(fused_results)
                continue
            trace = {
                "dense": [self._trace_item(chunk, rank) for rank, chunk in enumerate(dense_results, 1)],
                "bm25": [self._trace_item(chunk, rank) for rank, chunk in enumerate(sparse_results, 1)],
                "rrf": [self._trace_item(chunk, rank) for rank, chunk in enumerate(fused_results, 1)],
                "timing": {
                    "dense_start_ms": int((dense_started - batch_started) * 1000),
                    "dense_duration_ms": int(dense_duration * 1000),
                    "bm25_start_ms": int((sparse_started - batch_started) * 1000),
                    "bm25_duration_ms": int(sparse_duration * 1000),
                },
            }
            outputs.append((fused_results, trace))
        return outputs

    @staticmethod
    def _matches_filter(chunk: RetrievedChunk, filter_dict: Dict[str, Any]) -> bool:
        return all(chunk.metadata.get(key) == value for key, value in filter_dict.items())

    @classmethod
    def _trace_item(cls, chunk: RetrievedChunk, rank: int) -> Dict[str, Any]:
        return {
            "chunk_id": cls._chunk_key(chunk),
            "stable_chunk_id": stable_chunk_id(chunk.content, chunk.metadata),
            "rank": rank,
            "score": float(chunk.score),
            "source_file": chunk.metadata.get("source_file"),
        }

    @staticmethod
    def _chunk_key(chunk: RetrievedChunk) -> str:
        """优先使用向量库 ID；旧数据缺失时使用内容哈希作为稳定键。"""
        chunk_id = chunk.metadata.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        return "sha256-" + hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()[:24]

    def _rrf_fusion(
        self,
        dense_results: List[RetrievedChunk],
        sparse_results: List[RetrievedChunk],
        top_k: int,
        k: int = 60
    ) -> List[RetrievedChunk]:
        """
        Reciprocal Rank Fusion

        对两种检索的结果列表，按排名位置计算融合分数：
            rrf_score = 1/(k + dense_rank) + 1/(k + sparse_rank)

        出现在两种检索结果中的文档会获得更高的融合分数。
        """
        from collections import defaultdict

        scores = defaultdict(float)
        chunks = {}

        # 记录 Dense 排名
        for rank, chunk in enumerate(dense_results):
            key = self._chunk_key(chunk)
            scores[key] += 1.0 / (k + rank + 1)
            chunks[key] = chunk

        # 记录 Sparse 排名
        for rank, chunk in enumerate(sparse_results):
            key = self._chunk_key(chunk)
            scores[key] += 1.0 / (k + rank + 1)
            if key not in chunks:
                chunks[key] = chunk

        # 按 RRF 分数排序
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果
        result = []
        for chunk_id, score in sorted_items[:top_k]:
            chunk = chunks[chunk_id]
            metadata = {**chunk.metadata, "chunk_id": chunk_id}
            result.append(RetrievedChunk(
                content=chunk.content,
                metadata=metadata,
                score=score  # 这里 score 是 RRF 融合分数
            ))

        return result
