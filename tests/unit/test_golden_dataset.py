import json
from pathlib import Path

from tests.benchmark.validate_golden import build_catalog, validate_questions


ROOT = Path(__file__).resolve().parents[2]


def test_committed_golden_dataset_matches_versioned_corpus():
    questions = json.loads(
        (ROOT / "tests/benchmark/questions.json").read_text(encoding="utf-8")
    )
    catalog = build_catalog(ROOT / "tests/corpus")

    assert validate_questions(questions, catalog, expected_count=100) == []
