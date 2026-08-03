# Benchmark Attestations

完整盲测只在最终代码 Commit 上运行一次。评测通过后生成并提交
`current.json`；文件只包含指标摘要、配置指纹和签名，不包含问题、答案、
检索原文或 API Key。

```bash
# 1. 先提交最终代码，并保证工作树干净、API 与测试知识库已经启动。
python tests/benchmark/benchmark.py \
  --endpoint http://127.0.0.1:8000 \
  --split blind_test \
  --version "local-$(git rev-parse --short HEAD)" \
  --concurrency 5 \
  --timeout 180

# 2. 只有完整报告通过当前 Baseline Gate 才能生成签名证明。
python tests/benchmark/attestation.py create

# 3. 只提交精简证明；PR 不会再次调用模型。
git add tests/benchmark/attestations/current.json
git commit -m "test: attest blind regression benchmark"
```

本机私钥默认位于 `.benchmark/attestation_private_key.pem`，已被 `.gitignore`
排除。`tests/benchmark/attestation_public_key.pem` 只用于 CI 验签，可以公开。

首次签名证明不存在时，PR workflow 会明确报告 Gate 尚未启用但不会阻止基础设施
PR。只有完整盲测达到绝对质量下限并提交 `current.json` 后，才应把
`blind-regression` 配置为 `main` 的必需检查。
