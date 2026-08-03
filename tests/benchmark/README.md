# Regression Benchmark

该目录保存可复现、相互隔离的 RAG 回归评测资产。

## 数据分片

| 分片 | 数量 | 用途 | 允许调参 |
|---|---:|---|---|
| `train_dev` | 37 | 开发与问题定位 | 是 |
| `calibration` | 62 | 检索、Rerank、拒答阈值校准 | 仅限校准 |
| `blind_test` | 111 | 最终回归闸门 | 否 |

旧版 100 题及其基线保存在 `archive/` 与 `baselines/archive/`。

## 标准流程

```bash
# 1. 基础版本与 v2 版本分开入库，避免重复上传
python scripts/ingest_demo_corpus.py
python scripts/ingest_v2_corpus.py

# 2. 校验 210 题、分片隔离、证据与稳定 Chunk ID
python tests/benchmark/validate_golden.py --all-splits

# 3. 仅在 calibration 上确定阈值
python tests/benchmark/benchmark.py --split calibration --version calibration-v1
python tests/benchmark/calibrate_thresholds.py

# 4. 应用阈值并重启服务后，完整运行盲测
python tests/benchmark/benchmark.py --split blind_test --version v2.0

# 5. 首次建分层基线；完整 111 题、Smoke、溯源和绝对下限缺一不可
python tests/benchmark/regression_gate.py --bootstrap --promote

# 6. 后续回归：同一份报告同时检查质量和性能基线
python tests/benchmark/benchmark.py --split blind_test --version v2.1
python tests/benchmark/regression_gate.py
```

`benchmark.py` 默认运行 `blind_test`，并先执行 5 类 Smoke Gate；冒烟失败时不会
启动完整回归。`--promote-baseline` 和 `--fail-on-regression` 只能用于
`blind_test`，防止开发集或校准集污染最终基线。

## 主要文件

- `benchmark.py`：调用真实 SSE 链路，输出总览以及维度/难度分项指标。
- `calibrate_thresholds.py`：读取 calibration 报告，以显式路由标签和 OOD 标签校准阈值。
- `resolve_chunks.py`：复用生产 Loader/Splitter 解析或校验稳定 Chunk ID。
- `validate_golden.py`：校验 schema、证据、Chunk 标签、分片数量及跨分片重复。
- `regression_thresholds.json`：自动回归门禁。
- `regression_profiles.json`：质量/性能两层门槛；Recall@10 的绝对下限为 97%。
- `regression_gate.py`：校验 Dataset Hash、Git Commit、模型和阈值指纹后执行分层门禁。
- `pricing.json`：按供应商真实账单维护的每百万 Token 单价。
- `baselines/quality/latest.json`、`baselines/performance/latest.json`：相互独立的版本绑定基线。

GitHub Actions 的 `regression-gate.yml` 对检索、路由、模型和评测代码的 PR 自动运行。
仓库需配置 `RAG_BENCHMARK_ENDPOINT` Secret，指向与 PR Commit 对应的隔离评测部署，
并把 `blind-regression` 设为 `main` 的必需状态检查，才会真正阻止合并。

只有在111个盲测请求完整结束、知识库版本与配置正确、失败样本已经人工审查时，
才应提升 baseline。Bootstrap 还要求 Git worktree 为 clean，确保记录的 Commit 能
唯一复现实际运行代码。盲测结果只能用于接受或拒绝改动，不能继续反向调整阈值。
