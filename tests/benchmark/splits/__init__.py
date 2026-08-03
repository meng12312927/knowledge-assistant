"""Golden Dataset 分片加载器。

``train_dev`` 用于开发调试，``calibration`` 只能用于阈值校准，
``blind_test`` 只用于最终回归闸门。调用方不应使用盲测结果继续调参。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

_SPLITS_DIR = Path(__file__).resolve().parent

SPLIT_PATHS: Dict[str, Path] = {
    "train_dev": _SPLITS_DIR / "train_dev.json",
    "calibration": _SPLITS_DIR / "calibration.json",
    "blind_test": _SPLITS_DIR / "blind_test.json",
}

SPLIT_ORDER = ("train_dev", "calibration", "blind_test")
EXPECTED_SPLIT_COUNTS = {
    "train_dev": 37,
    "calibration": 62,
    "blind_test": 111,
}

LEGACY_PATH = _SPLITS_DIR.parent / "questions.json"


def load_split(name: str) -> List[Dict[str, Any]]:
    """Load a named split. Raises FileNotFoundError if missing."""
    if name == "all":
        questions: List[Dict[str, Any]] = []
        for split_name in SPLIT_ORDER:
            path = SPLIT_PATHS.get(split_name)
            if path and path.exists():
                questions.extend(json.loads(path.read_text(encoding="utf-8")))
        return questions
    if name == "legacy":
        if not LEGACY_PATH.exists():
            raise FileNotFoundError(f"Legacy dataset not found: {LEGACY_PATH}")
        return json.loads(LEGACY_PATH.read_text(encoding="utf-8"))
    path = SPLIT_PATHS.get(name)
    if path is None:
        raise ValueError(
            f"Unknown split '{name}'. "
            f"Valid: {sorted(SPLIT_PATHS)} | all | legacy"
        )
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_questions(path: Path) -> List[Dict[str, Any]]:
    """Load questions from an explicit path (backward-compatible)."""
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_questions(
    split: str | None = None,
    questions_path: Path | None = None,
) -> List[Dict[str, Any]]:
    """Resolve questions from split name or explicit path.

    Priority: explicit path > split name > default blind_test
    """
    if questions_path is not None:
        return load_questions(questions_path)
    return load_split(split or "blind_test")
