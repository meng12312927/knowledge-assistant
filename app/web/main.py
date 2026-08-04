"""
Streamlit 前端界面（v0.2）

支持：
- 文件上传（multipart/form-data）
- 多轮对话历史
- 查看引用来源
- 删除已上传文档

启动命令：
    streamlit run app/web/main.py
"""

import streamlit as st
import requests
import json
import os
from typing import List

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="企业制度与员工手册助手",
    page_icon="📘",
    layout="wide"
)


def format_trace_metric(value, unit: str = ""):
    """格式化 Trace 概览指标，区分缺失值与真实的零值。"""
    if value is None:
        return "N/A"
    return f"{value} {unit}" if unit else value


def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())


def load_history():
    """从后端加载对话历史"""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/api/v1/chat/history",
            params={"session_id": st.session_state.session_id},
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            old_messages = st.session_state.messages.copy()
            new_messages = []
            for msg in data.get("messages", []):
                role = msg.get("role")
                content = msg.get("content")
                if not role or not content:
                    continue
                stored_sources = msg.get("sources", []) or []
                stored_citations = (
                    stored_sources
                    if stored_sources and stored_sources[0].get("citation_id")
                    else []
                )
                new_messages.append({
                    "role": role,
                    "content": content,
                    "sources": [] if stored_citations else stored_sources,
                    "citations": stored_citations,
                    "citation_verification": msg.get("verification"),
                    "trace": msg.get("trace"),
                })
            st.session_state.messages = new_messages
    except Exception as e:
        st.sidebar.warning(f"加载历史失败: {e}")


def call_chat_api(query: str) -> dict:
    """调用后端问答 API（非流式）"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/chat",
            json={
                "query": query,
                "session_id": st.session_state.session_id,
                "top_k": 15,
                "strict_verification": st.session_state.get("strict_verification"),
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ 无法连接到后端服务。请先启动 API：\n```\nuvicorn app.api.main:app --port 8000\n```")
        return None
    except Exception as e:
        st.error(f"❌ 请求失败: {str(e)}")
        return None


def stream_chat_api(query: str):
    """调用后端流式问答 API，yield token 内容"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/chat/stream",
            json={
                "query": query,
                "session_id": st.session_state.session_id,
                "top_k": 15,
                "strict_verification": st.session_state.get("strict_verification"),
            },
            stream=True,
            timeout=300
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = json.loads(line[6:])
                msg_type = data.get('type')
                if msg_type == 'token':
                    yield data['content']
                elif msg_type == 'sources':
                    st.session_state.last_sources = data.get('sources', [])
                elif msg_type == 'citations':
                    st.session_state.last_citations = data.get('citations', [])
                elif msg_type == 'citation_verification':
                    st.session_state.last_citation_verification = data.get('verification')
                elif msg_type == 'rag_trace':
                    st.session_state.last_rag_trace = data.get('trace')
                elif msg_type == 'error':
                    st.session_state.stream_error = data.get('message', '未知错误')
                    break
                elif msg_type == 'done':
                    break
    except requests.exceptions.ConnectionError:
        st.error("❌ 无法连接到后端服务。请先启动 API：\n```\nuvicorn app.api.main:app --port 8000\n```")
    except Exception as e:
        st.error(f"❌ 请求失败: {str(e)}")


def upload_file(file) -> dict:
    """上传文件到后端"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/ingest",
            files={"file": (file.name, file.getvalue(), file.type)},
            timeout=180
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"上传失败: {e}")
        return None


def render_citations(citations: List[dict]):
    """展示 `[S1] -> chunk_id -> 原文` 的可验证引用链。"""
    if not citations:
        return
    with st.expander("📚 查看结构化引用"):
        for citation in citations:
            citation_id = citation.get("citation_id", "S?")
            source_file = citation.get("source_file", "未知文件")
            page = citation.get("page_number") or "N/A"
            chunk_id = citation.get("chunk_id", "N/A")
            st.markdown(f"### [{citation_id}] {source_file} · 第{page}页")
            st.code(chunk_id, language=None)
            st.markdown(f"> {citation.get('content', '').replace(chr(10), chr(10) + '> ')}")
            st.caption(f"检索分数：{float(citation.get('score') or 0):.4f}")


def render_citation_verification(verification: dict):
    """展示答案结论与引用原文的一致性审计结果。"""
    if not verification:
        return
    status = verification.get("status", "unverified")
    message = verification.get("message", "")
    if status == "verified":
        st.success(f"✅ Citation Verification：已通过 · {message}")
    elif status == "skipped":
        st.info(f"ℹ️ Citation Verification：已跳过 · {message}")
    elif status == "failed":
        st.error(f"❌ Citation Verification：未通过 · {message}")
    else:
        st.warning(f"⚠️ Citation Verification：未完成 · {message}")

    items = verification.get("items", [])
    if items:
        with st.expander("🔍 查看逐结论核验"):
            for item in items:
                refs = " ".join(f"[{value}]" for value in item.get("citation_ids", [])) or "无引用"
                st.markdown(f"**{item.get('verdict', 'unknown')}** · {refs}  ")
                st.markdown(item.get("claim", ""))
                if item.get("reason"):
                    st.caption(item["reason"])


def render_rag_trace(trace: dict):
    """展示 Query 改写到 Citation Verification 的六阶段可观测轨迹。"""
    if not trace:
        return
    with st.expander("🧭 查看 RAG 全流程追踪"):
        metric_columns = st.columns(4)
        metric_columns[0].metric("知识库版本", format_trace_metric(trace.get("knowledge_base_version")))
        metric_columns[1].metric("TTFT", format_trace_metric(trace.get("ttft_ms"), "ms"))
        metric_columns[2].metric("总延迟", format_trace_metric(trace.get("total_latency_ms"), "ms"))
        metric_columns[3].metric(
            "总 Token",
            format_trace_metric((trace.get("token_usage") or {}).get("total_tokens")),
        )

        ttft_columns = st.columns(4)
        ttft_columns[0].metric("模型 TTFT", format_trace_metric(trace.get("generation_ttft_ms"), "ms"))
        ttft_columns[1].metric(
            "模型首 Token 时刻",
            format_trace_metric(trace.get("generation_first_token_at_ms"), "ms"),
        )
        ttft_columns[2].metric("核验完成", format_trace_metric(trace.get("verified_ttft_ms"), "ms"))
        ttft_columns[3].metric("SSE 完成", format_trace_metric(trace.get("sse_total_latency_ms"), "ms"))

        st.markdown("#### Query 路由与缓存")
        st.json({
            "strategy": trace.get("query_strategy"),
            "multiquery_triggered": trace.get("multiquery_triggered"),
            "reason": trace.get("multiquery_reason"),
            "cache_hits": trace.get("cache_hits", {}),
            "cache_stats": trace.get("cache_stats", {}),
        })

        st.markdown("#### Trace Spans")
        st.dataframe(trace.get("spans", []), use_container_width=True, hide_index=True)

        st.markdown("#### 版本信息")
        st.markdown(f"知识库：`{trace.get('knowledge_base_version', 'N/A')}`")
        st.json(trace.get("document_versions", {}))

        st.markdown("#### Token Usage")
        st.json(trace.get("token_usage", {}))

        st.markdown("#### 1. Query 改写结果")
        for index, query in enumerate(trace.get("query_variants", []), 1):
            st.markdown(f"{index}. `{query}`")

        st.markdown("#### 2. Dense / BM25 原始排名")
        dense_tab, bm25_tab = st.tabs(["Dense", "BM25"])
        with dense_tab:
            st.dataframe(trace.get("dense_rankings", []), use_container_width=True, hide_index=True)
        with bm25_tab:
            st.dataframe(trace.get("bm25_rankings", []), use_container_width=True, hide_index=True)

        st.markdown("#### 3. RRF / Rerank 分数")
        rrf_tab, rerank_tab = st.tabs(["RRF", "Rerank"])
        with rrf_tab:
            st.dataframe(trace.get("rrf_rankings", []), use_container_width=True, hide_index=True)
        with rerank_tab:
            st.dataframe(trace.get("rerank_rankings", []), use_container_width=True, hide_index=True)

        st.markdown("#### 4. selected_chunk_ids")
        st.code("\n".join(trace.get("selected_chunk_ids", [])) or "无", language=None)

        if trace.get("subquestions"):
            st.markdown("#### 子问题证据覆盖")
            st.metric(
                "Evidence Coverage",
                f"{float(trace.get('evidence_coverage') or 0):.1%}",
            )
            st.dataframe(
                trace.get("subquestions", []),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### 5. [S1] → chunk_id 映射")
        citation_map = trace.get("citation_map", {})
        st.code("\n".join(f"[{key}] -> {value}" for key, value in citation_map.items()) or "无", language=None)

        st.markdown("#### 6. Citation Verification 结果")
        verification = trace.get("citation_verification") or {}
        st.json(verification)


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("⚙️ 设置")

        # 加载历史
        if st.button("🔄 加载历史对话"):
            load_history()
            st.rerun()

        if st.button("🗑️ 清空当前对话"):
            st.session_state.messages = []
            try:
                requests.delete(
                    f"{API_BASE_URL}/api/v1/chat/history",
                    params={"session_id": st.session_state.session_id}
                )
            except:
                pass
            st.rerun()

        st.divider()

        # 引用核验模式
        st.subheader("🔍 引用核验")
        if "strict_verification" not in st.session_state:
            st.session_state.strict_verification = True
        st.session_state.strict_verification = st.toggle(
            "严格核验模式",
            value=st.session_state.strict_verification,
            help="开启：核验失败拒答。关闭：答案照发，核验报告仅作参考",
        )

        st.divider()

        # 知识库统计
        st.subheader("📊 知识库状态")
        if st.button("刷新统计"):
            try:
                resp = requests.get(f"{API_BASE_URL}/api/v1/stats", timeout=5)
                if resp.status_code == 200:
                    stats = resp.json()
                    st.metric("向量库文档数", stats.get("total_documents_in_store", 0))
                    st.metric("已上传文件数", stats.get("total_files_uploaded", 0))
            except Exception as e:
                st.warning(f"服务未启动: {e}")

        st.divider()

        # 已上传文档列表
        st.subheader("📁 已上传文档")
        try:
            resp = requests.get(f"{API_BASE_URL}/api/v1/documents", timeout=5)
            if resp.status_code == 200:
                docs = resp.json().get("documents", [])
                for doc in docs:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.caption(f"{doc['filename']} · v{doc.get('version', 1)} ({doc['chunk_count']} chunks)")
                    with col2:
                        if st.button("🗑️", key=f"del_{doc['doc_id']}"):
                            try:
                                requests.delete(
                                    f"{API_BASE_URL}/api/v1/documents/{doc['doc_id']}",
                                    timeout=10
                                )
                                st.rerun()
                            except Exception as e:
                                st.error(f"删除失败: {e}")
        except:
            st.caption("暂无文档")

        st.divider()

        # 危险操作：一键清空数据库
        with st.expander("⚠️ 危险操作：一键清空数据库", expanded=False):
            st.error("此操作将永久删除所有数据，包括：\n- 向量库中的所有文档 chunk\n- SQLite 中的文档元数据\n- 所有对话历史\n\n**不可恢复！**")
            confirm_text = st.text_input(
                "请输入 `CLEAR` 以确认清空",
                placeholder="输入 CLEAR",
                key="clear_db_confirm"
            )
            if st.button("🗑️ 确认清空全部数据", type="primary", key="btn_clear_db"):
                if confirm_text.strip().upper() == "CLEAR":
                    try:
                        resp = requests.post(
                            f"{API_BASE_URL}/api/v1/database/clear",
                            timeout=30
                        )
                        if resp.status_code == 200:
                            st.success("数据库已清空！")
                            # 重置前端状态
                            st.session_state.messages = []
                            import uuid
                            st.session_state.session_id = str(uuid.uuid4())
                            st.rerun()
                        else:
                            st.error(f"清空失败: {resp.text}")
                    except Exception as e:
                        st.error(f"请求失败: {e}")
                else:
                    st.warning("请输入 `CLEAR` 以确认操作")

        st.divider()

        # 文件上传（支持批量）
        st.subheader("⬆️ 上传新文档")
        uploaded_files = st.file_uploader(
            "选择文件（可多选）",
            type=["pdf", "docx", "pptx", "txt", "md", "html"],
            accept_multiple_files=True,
            help="支持 PDF、Word、PPT、Markdown 等格式，可一次选择多个文件批量上传"
        )

        # 显示已选择的文件列表
        if uploaded_files:
            st.caption(f"已选择 {len(uploaded_files)} 个文件：")
            for f in uploaded_files:
                size_mb = len(f.getvalue()) / 1024 / 1024
                st.caption(f"  • {f.name} ({size_mb:.1f} MB)")

        if uploaded_files and st.button("🚀 开始批量摄取"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, file in enumerate(uploaded_files):
                progress = (i) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"[{i+1}/{len(uploaded_files)}] 正在处理: {file.name}...")

                result = upload_file(file)
                if result:
                    results.append(result)
                else:
                    results.append({"filename": file.name, "error": "上传失败"})

            progress_bar.progress(1.0)
            status_text.empty()

            # 显示汇总结果
            success_count = sum(1 for r in results if "error" not in r)
            update_count = sum(1 for r in results if r.get("is_update") and "error" not in r)
            new_count = success_count - update_count

            st.success(f"📊 批量摄取完成！成功 {success_count}/{len(uploaded_files)} 个")
            if new_count:
                st.caption(f"  • 新增: {new_count} 个")
            if update_count:
                st.caption(f"  • 覆盖: {update_count} 个")

            # 显示每个文件的详细结果
            with st.expander("📋 详细结果"):
                for r in results:
                    if "error" in r:
                        st.error(f"❌ {r['filename']}: {r['error']}")
                    else:
                        action = "覆盖" if r.get("is_update") else "新增"
                        st.caption(f"✅ [{action}] {r['filename']} → {r['chunks_ingested']} chunks")

            # 自动刷新文档列表
            st.rerun()

        st.divider()
        st.caption("Employee Policy Assistant v1.0.0 · 合成演示数据")


def render_chat_interface():
    """渲染聊天界面"""
    st.title("📘 企业制度与员工手册助手")
    st.caption("查询休假、报销、远程办公、信息安全、绩效与入离职制度；回答附原文来源")

    suggested_prompt = None
    if not st.session_state.messages:
        st.markdown("##### 试试这些问题")
        suggestions = [
            "预计差旅费用15000元需要谁审批？",
            "极端天气临时远程办公占每周额度吗？",
            "发现账号异常登录后多久内报告？",
            "购买1800元岗位课程需要哪些审批？",
        ]
        columns = st.columns(2)
        for index, suggestion in enumerate(suggestions):
            if columns[index % 2].button(suggestion, key=f"suggestion_{index}", use_container_width=True):
                suggested_prompt = suggestion

    # 显示对话历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("citation_verification"):
                render_citation_verification(message["citation_verification"])
            if message["role"] == "assistant" and message.get("trace"):
                render_rag_trace(message["trace"])
            if message["role"] == "assistant" and message.get("citations"):
                render_citations(message["citations"])
            elif message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📚 查看来源"):
                    for i, source in enumerate(message["sources"][:5], 1):
                        meta = source.get("metadata", {})
                        st.markdown(f"**[{i}]** {source.get('content', '')[:200]}...")
                        st.caption(
                            f"来源: {meta.get('source_file', 'N/A')} | "
                            f"页码: {meta.get('page_number', 'N/A')} | "
                            f"相关度: {source.get('score', 0):.3f}"
                        )

    # 用户输入
    prompt = st.chat_input("例如：预计差旅费用15000元需要谁审批？") or suggested_prompt
    if prompt:
        st.session_state.pop("last_sources", None)
        st.session_state.pop("last_citations", None)
        st.session_state.pop("last_citation_verification", None)
        st.session_state.pop("last_rag_trace", None)
        st.session_state.pop("stream_error", None)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # 流式输出
            answer = st.write_stream(
                stream_chat_api(prompt)
            )

            stream_error = st.session_state.pop("stream_error", None)
            if stream_error:
                st.error(f"回答失败：{stream_error}")
                answer = answer or f"回答失败：{stream_error}"

            sources = st.session_state.get("last_sources", [])
            citations = st.session_state.get("last_citations", [])
            verification = st.session_state.get("last_citation_verification")
            trace = st.session_state.get("last_rag_trace")
            render_citation_verification(verification)
            render_rag_trace(trace)
            if citations:
                render_citations(citations)
            elif sources:
                with st.expander("📚 查看来源"):
                    for i, source in enumerate(sources[:5], 1):
                        meta = source.get("metadata", {})
                        st.markdown(f"**[{i}]** {source.get('content', '')[:200]}...")
                        st.caption(
                            f"来源: {meta.get('source_file', 'N/A')} | "
                            f"页码: {meta.get('page_number', 'N/A')} | "
                            f"相关度: {float(source.get('score') or 0):.3f}"
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "citations": citations,
                "citation_verification": verification,
                "trace": trace,
            })


def main():
    init_session_state()
    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()
