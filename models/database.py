"""
SQLite 持久化层

管理：
1. 对话历史（conversations 表）
2. 文档元数据（documents 表，用于去重和删除）
"""

import json
import sqlite3
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

# 串行化 SQLite 写入，避免 WAL 模式下并发写锁
_db_write_lock = threading.Lock()


class _JSONEncoder(json.JSONEncoder):
    """处理不可 JSON 序列化的类型（如 datetime、set 等）"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, set):
            return list(obj)
        try:
            return str(obj)
        except Exception:
            return repr(obj)

DB_PATH = Path("./data/app.db")


def _get_conn() -> sqlite3.Connection:
    """获取数据库连接"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库表"""
    conn = _get_conn()
    try:
        cursor = conn.cursor()

        # 对话历史
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                verification TEXT,
                trace TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        columns = {row[1] for row in cursor.execute("PRAGMA table_info(conversations)").fetchall()}
        if "verification" not in columns:
            cursor.execute("ALTER TABLE conversations ADD COLUMN verification TEXT")
        if "trace" not in columns:
            cursor.execute("ALTER TABLE conversations ADD COLUMN trace TEXT")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_session
            ON conversations(session_id, created_at)
        """)

        # 文档元数据（用于去重和删除）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                version INTEGER NOT NULL DEFAULT 1,
                content_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        document_columns = {row[1] for row in cursor.execute("PRAGMA table_info(documents)").fetchall()}
        if "version" not in document_columns:
            cursor.execute("ALTER TABLE documents ADD COLUMN version INTEGER NOT NULL DEFAULT 1")
        if "content_hash" not in document_columns:
            cursor.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT")
        cursor.execute("UPDATE documents SET content_hash = doc_id WHERE content_hash IS NULL")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "INSERT OR IGNORE INTO system_metadata (key, value) VALUES ('knowledge_base_version', '1')"
        )

        conn.commit()
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════
# 对话历史操作
# ═══════════════════════════════════════════════════════════

def save_message(
    session_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    verification: Optional[Dict[str, Any]] = None,
    trace: Optional[Dict[str, Any]] = None,
) -> int:
    """保存一条对话消息（线程安全）"""
    with _db_write_lock:
        conn = _get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO conversations (session_id, role, content, sources, verification, trace) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    role,
                    content,
                    json.dumps(sources, cls=_JSONEncoder, ensure_ascii=False) if sources else None,
                    json.dumps(verification, cls=_JSONEncoder, ensure_ascii=False) if verification else None,
                    json.dumps(trace, cls=_JSONEncoder, ensure_ascii=False) if trace else None,
                )
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()


def get_conversation_history(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """获取某个 session 的对话历史"""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content, sources, verification, trace, created_at FROM conversations WHERE session_id = ? ORDER BY created_at LIMIT ?",
            (session_id, limit)
        )
        rows = cursor.fetchall()

        result = []
        for row in rows:
            sources = json.loads(row["sources"]) if row["sources"] else None
            result.append({
                "role": row["role"],
                "content": row["content"],
                "sources": sources,
                "verification": json.loads(row["verification"]) if row["verification"] else None,
                "trace": json.loads(row["trace"]) if row["trace"] else None,
                "created_at": row["created_at"]
            })
        return result
    finally:
        conn.close()


def clear_conversation(session_id: str) -> None:
    """清空某个 session 的对话历史（线程安全）"""
    with _db_write_lock:
        conn = _get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()


# ═══════════════════════════════════════════════════════════
# 文档元数据操作
# ═══════════════════════════════════════════════════════════

def save_document_meta(doc_id: str, filename: str, chunk_count: int = 0, version: int = 1) -> None:
    """保存/更新文档元数据（线程安全）"""
    with _db_write_lock:
        conn = _get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO documents (doc_id, filename, chunk_count, version, content_hash)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    filename=excluded.filename,
                    chunk_count=excluded.chunk_count,
                    version=excluded.version,
                    content_hash=excluded.content_hash,
                    created_at=CURRENT_TIMESTAMP
                """,
                (doc_id, filename, chunk_count, version, doc_id)
            )
            conn.commit()
        finally:
            conn.close()


def get_document_meta(doc_id: str) -> Optional[Dict[str, Any]]:
    """获取文档元数据"""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()

        if row:
            return {
                "doc_id": row["doc_id"],
                "filename": row["filename"],
                "chunk_count": row["chunk_count"],
                "version": row["version"],
                "content_hash": row["content_hash"],
                "created_at": row["created_at"]
            }
        return None
    finally:
        conn.close()


def list_documents() -> List[Dict[str, Any]]:
    """列出所有已上传的文档"""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id, filename, chunk_count, version, content_hash, created_at FROM documents ORDER BY created_at DESC")
        rows = cursor.fetchall()

        return [
            {
                "doc_id": row["doc_id"],
                "filename": row["filename"],
                "chunk_count": row["chunk_count"],
                "version": row["version"],
                "content_hash": row["content_hash"],
                "created_at": row["created_at"]
            }
            for row in rows
        ]
    finally:
        conn.close()


def get_document_meta_by_filename(filename: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE filename = ? ORDER BY version DESC LIMIT 1",
            (filename,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_knowledge_base_version() -> int:
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT value FROM system_metadata WHERE key = 'knowledge_base_version'"
        ).fetchone()
        return int(row["value"]) if row else 1
    finally:
        conn.close()


def bump_knowledge_base_version() -> int:
    with _db_write_lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT value FROM system_metadata WHERE key = 'knowledge_base_version'"
            ).fetchone()
            current = int(row["value"]) if row else 1
            new_version = current + 1
            conn.execute(
                "UPDATE system_metadata SET value = ?, updated_at = CURRENT_TIMESTAMP WHERE key = 'knowledge_base_version'",
                (str(new_version),),
            )
            conn.commit()
            return new_version
        finally:
            conn.close()


def delete_document_meta(doc_id: str) -> None:
    """删除文档元数据记录（线程安全）"""
    with _db_write_lock:
        conn = _get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.commit()
        finally:
            conn.close()


def clear_all_data() -> None:
    """清空所有数据（线程安全）：文档元数据 + 对话历史"""
    with _db_write_lock:
        conn = _get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents")
            cursor.execute("DELETE FROM conversations")
            conn.commit()
        finally:
            conn.close()
