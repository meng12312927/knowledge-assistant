"""通过 API 导入 ``tests/corpus`` 下的版本化 v2 测试语料。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


VERSION_PATTERN = re.compile(r"_v(?P<version>\d+)$", flags=re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(description="导入 Golden Dataset v2 语料")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--corpus", default="tests/corpus")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    import requests

    files = [
        path
        for path in sorted(Path(args.corpus).rglob("*.txt"))
        if VERSION_PATTERN.search(path.stem)
    ]
    if not files:
        raise SystemExit(f"未找到版本化测试文档：{args.corpus}/**/*_vN.txt")

    succeeded = 0
    for path in files:
        declared_version = int(VERSION_PATTERN.search(path.stem).group("version"))
        with path.open("rb") as handle:
            response = requests.post(
                f"{args.api.rstrip('/')}/api/v1/ingest",
                files={"file": (path.name, handle, "text/plain")},
                timeout=args.timeout,
            )
        if not response.ok:
            print(f"[FAIL] {path.name}: {response.status_code} {response.text}")
            continue
        result = response.json()
        actual_version = result.get("document_version")
        if actual_version != declared_version:
            print(
                f"[FAIL] {path.name}: 声明 v{declared_version}，"
                f"服务端记录 v{actual_version}"
            )
            continue
        succeeded += 1
        print(
            f"[OK] {path.name}: version={actual_version} "
            f"chunks={result.get('chunks_ingested', 0)}"
        )

    print(f"v2 语料导入完成：{succeeded}/{len(files)}")
    if succeeded != len(files):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
