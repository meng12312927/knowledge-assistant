"""Stable logical identifiers for retrieval chunks.

Chroma's physical IDs are UUIDs and change when the same document is ingested
again. Regression datasets therefore must not use them as durable labels.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Mapping


def normalize_chunk_text(value: str) -> str:
    """Normalize inconsequential whitespace without changing document wording."""
    normalized = unicodedata.normalize("NFKC", value or "")
    return re.sub(r"\s+", " ", normalized).strip()


def stable_chunk_id(content: str, metadata: Mapping[str, Any] | None = None) -> str:
    """Return a reproducible ID based on source name and normalized chunk text."""
    metadata = metadata or {}
    source_file = unicodedata.normalize(
        "NFKC", str(metadata.get("source_file") or "unknown")
    ).strip()
    payload = f"{source_file}\n{normalize_chunk_text(content)}".encode("utf-8")
    return "schunk-" + hashlib.sha256(payload).hexdigest()[:24]
