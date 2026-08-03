import json
from pathlib import Path

from tests.benchmark.splits import EXPECTED_SPLIT_COUNTS, load_split
from tests.benchmark.validate_golden import (
    build_catalog,
    validate_questions,
    validate_split_suite,
)


ROOT = Path(__file__).resolve().parents[2]


def test_committed_golden_dataset_matches_versioned_corpus():
    questions = json.loads(
        (ROOT / "tests/benchmark/questions.json").read_text(encoding="utf-8")
    )
    catalog = build_catalog(ROOT / "tests/corpus")

    assert validate_questions(questions, catalog, expected_count=100) == []


def test_split_golden_dataset_is_isolated_and_matches_corpus():
    catalog = build_catalog(ROOT / "tests/corpus")

    assert {name: len(load_split(name)) for name in EXPECTED_SPLIT_COUNTS} == (
        EXPECTED_SPLIT_COUNTS
    )
    assert validate_split_suite(catalog) == []
