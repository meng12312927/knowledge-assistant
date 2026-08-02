"""检查企业制度演示语料是否完整。

语料是虚构的“星云科技”内部制度，仅用于 RAG 演示，不对应真实公司。
制度正文直接版本化在 tests/corpus，避免生成脚本与测试材料出现偏差。
"""

from pathlib import Path


EXPECTED_FILES = {
    "hr/星云科技员工手册.txt",
    "hr/考勤与休假管理制度.txt",
    "hr/入职转正与离职交接制度.txt",
    "finance/差旅与费用报销制度.txt",
    "it/信息安全与数据分级制度.txt",
    "workplace/远程办公管理办法.txt",
    "performance/绩效管理制度.txt",
    "benefits/员工福利与学习发展制度.txt",
}


def main():
    corpus_dir = Path(__file__).parent / "corpus"
    existing = {
        str(path.relative_to(corpus_dir))
        for path in corpus_dir.rglob("*.txt")
    }
    missing = EXPECTED_FILES - existing
    unexpected = existing - EXPECTED_FILES
    if missing or unexpected:
        raise SystemExit(
            f"语料不一致 missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    print(f"企业制度语料检查通过：{len(existing)} 个文档")


if __name__ == "__main__":
    main()
