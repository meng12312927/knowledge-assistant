"""通过 API 批量导入虚构的星云科技制度语料。"""

import argparse
import re
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--corpus", default="tests/corpus")
    parser.add_argument(
        "--include-versioned",
        action="store_true",
        help="同时导入 *_v2.txt 等版本化语料；默认只导入基础版本",
    )
    args = parser.parse_args()

    import requests

    files = sorted(Path(args.corpus).rglob("*.txt"))
    if not args.include_versioned:
        files = [
            path
            for path in files
            if not re.search(r"_v\d+$", path.stem, flags=re.IGNORECASE)
        ]
    if not files:
        raise SystemExit(f"未找到演示文档：{args.corpus}")

    succeeded = 0
    for path in files:
        with path.open("rb") as handle:
            response = requests.post(
                f"{args.api.rstrip('/')}/api/v1/ingest",
                files={"file": (path.name, handle, "text/plain")},
                timeout=180,
            )
        if response.ok:
            result = response.json()
            succeeded += 1
            print(f"[OK] {path.name}: {result.get('chunks_ingested', 0)} chunks")
        else:
            print(f"[FAIL] {path.name}: {response.status_code} {response.text}")

    print(f"导入完成：{succeeded}/{len(files)}")
    if succeeded != len(files):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
