# Regression Benchmark

该目录保存可复现的 RAG 回归评测资产：

- `questions.json`：固定 100 题 Golden Dataset。
- `benchmark.py`：调用真实 SSE 问答链，计算检索、生成、延迟和成本指标。
- `validate_golden.py`：校验 schema、稳定 chunk 标签和逐字证据。
- `generate_golden.py`：仅在语料变化时生成待人工评审的候选题。
- `regression_thresholds.json`：自动回归门禁。
- `pricing.json`：按实际供应商账单维护的每百万 Token 单价。
- `baselines/latest.json`：经过人工确认后提升的最近基准。
- `results/`：当前运行报告。

日常流程：

```text
修改代码
  → 运行固定 5 类 Smoke Gate
  → 校验 Golden Dataset
  → 运行固定 100 题
  → 生成 JSON / Markdown 报告
  → 与 Baseline 比较并执行回归门禁
```

直接运行 `benchmark.py` 时 Smoke Gate 会自动执行；5/5 通过后才会开始 100 题。
单独排查冒烟失败可运行：

```bash
python tests/smoke/smoke_test.py
```

固定用例和断言位于 `tests/smoke/cases.json`，失败报告位于
`tests/smoke/results/smoke_report.md`。`--skip-smoke` 只保留给诊断场景。

建议只有在以下条件同时满足时才使用 `--promote-baseline`：

1. 100 个请求全部完成，服务配置与目标环境一致；
2. Golden Dataset 的版本和知识库版本符合预期；
3. 报告中的失败样本已经人工审查；
4. 指标变化是有意且可以解释的。
