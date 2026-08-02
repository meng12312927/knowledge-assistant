from rag.chunk_identity import stable_chunk_id


def test_stable_chunk_id_ignores_whitespace_and_runtime_metadata():
    first = stable_chunk_id(
        "第一条  员工应遵守制度。\n第二条 生效。",
        {"source_file": "员工手册.txt", "chunk_id": "random-a", "page": 1},
    )
    second = stable_chunk_id(
        " 第一条 员工应遵守制度。 第二条 生效。 ",
        {"source_file": "员工手册.txt", "chunk_id": "random-b", "page": 99},
    )
    assert first == second


def test_stable_chunk_id_changes_with_source_or_content():
    base = stable_chunk_id("年假为五天。", {"source_file": "员工手册.txt"})
    assert base != stable_chunk_id("年假为十天。", {"source_file": "员工手册.txt"})
    assert base != stable_chunk_id("年假为五天。", {"source_file": "休假制度.txt"})
