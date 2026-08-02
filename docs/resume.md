# 简历与面试素材

## 推荐项目名称

企业制度与员工手册智能问答系统

## 简历描述

- 设计 Retrieval-first RAG 检索链路：所有 Query 先执行原始 Dense + BM25 检索，召回不足才由 DeepSeek 进行 MultiQuery；实现 Query Rewrite / Embedding LRU 缓存与 single-flight 防击穿，并将命中、等待和升级原因写入 Trace。
- 使用阿里云 `text-embedding-v4`、Dense + BM25 双路召回和两层 RRF 融合，多 Query 的 Dense/BM25 通道通过共享线程池并行执行；召回最多 40 个候选后调用 `qwen3-rerank`，筛选最多 6 个证据块进入生成上下文。
- 建立 `[S1] → chunk_id → 原文` 结构化证据链和严格 Citation Verification；核验失败时 fail-closed，并记录 Query 排名、Rerank 分数、selected chunks、知识库/文档版本及完整阶段 span。
- 基于 FastAPI SSE 和 Streamlit 实现问答与可观测界面，分别记录生成模型 TTFT、核验后服务端用户可见 TTFT、客户端 `done` 延迟和 LLM/Reranker Token；50 次本地历史回归中平均 SSE 结束延迟降低 50.77%、P95 降低 46.99%、DeepSeek Token 降低 25.76%，技术错误率为 0%。
- 使用 pytest、GitHub Actions 和 Docker Compose 完成自动化验证与部署；性能脚本支持 50～100 次固定种子、预热、并发、P50/P95/P99、错误率、吞吐量和优化前后报告对比。

性能数字来自 `tests/results/performance_final_20260722.json`，该报告生成于最终 OOD 拒答阈值补丁前；面试时应说明这是单机本地历史快照，不是生产 SLA，且 50 个样本的 nearest-rank P99 等于最大异常值。

## 面试重点

1. 为什么所有 Query 都先直检，怎样通过首次 RRF 与 Reranker 结果决定 MultiQuery、Agent 升级或 OOD 拒答？
2. Dense 与 BM25 分别解决什么问题，两层 RRF 和并行召回如何实现？
3. 为什么先召回 40 个候选、经 `qwen3-rerank` 后只给生成模型 6 个？
4. 严格 Citation Verification 为什么必须在用户可见输出之前完成，失败策略是什么？
5. 生成 TTFT、核验完成、服务端 yield 和客户端收到 SSE `done` 为什么不能混为一个指标？
6. 当前本地缓存、SQLite 和 Chroma 如何演进为多副本、多租户的生产架构？

## 演示建议

准备四类问题：原始召回足够的普通问题、需要 MultiQuery 的低召回问题、工具/多步骤 Agent 问题、完全超出知识库的问题。演示 `direct / adaptive_fallback`、single-flight、Agent 决策、OOD 拒答、并行 span、Rerank 分数、`[S1] → chunk_id → 原文` 引用链，以及 Citation Verification 的通过与 fail-closed。
