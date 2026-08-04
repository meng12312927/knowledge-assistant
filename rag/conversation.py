"""轻量级多轮对话管理器。

滑动窗口 + ContextCompressor 压缩：
- 最近 N 轮完整保留在 prompt 中
- 超出窗口的旧轮次用已有的 ContextCompressor 压缩为摘要
- 摘要按 session_id 缓存，旧轮次不变时不重新压缩
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from models.database import get_conversation_history


class ConversationManager:
    """多轮对话上下文管理器。"""

    def __init__(
        self,
        llm,
        *,
        max_window_turns: int = 6,
        compress_threshold: int = 10,
    ):
        """
        Args:
            llm: LLM 客户端（用于压缩旧对话）
            max_window_turns: 滑动窗口保留的最近对话轮数
            compress_threshold: 超过此轮数时触发压缩
        """
        self.llm = llm
        self.max_window_turns = max(2, max_window_turns)
        self.compress_threshold = max(
            max_window_turns + 2, compress_threshold
        )
        self._summaries: Dict[str, str] = {}
        # 缓存：上次压缩时的旧轮次哈希 → 摘要
        self._cache: Dict[str, str] = {}
        # 每个 session 的轮次计数（用于判断是否需要重新压缩）
        self._last_old_turn_count: Dict[str, int] = {}

    def build_context(self, session_id: str, current_query: str) -> str:
        """构建注入 prompt 的对话上下文。

        Returns:
            格式化的上下文字符串，无历史时返回空字符串。
        """
        if not session_id:
            return ""

        history = get_conversation_history(session_id)
        if not history:
            return ""

        total_turns = len(history)
        if total_turns <= self.max_window_turns:
            # 全在窗口内，不压缩
            return self._format_recent(history, current_query)

        # 拆分：旧轮次（压缩） + 最近轮次（原文）
        old_turns = history[: total_turns - self.max_window_turns]
        recent_turns = history[-self.max_window_turns:]

        # 只在新旧轮次变化时重新压缩
        old_count = len(old_turns)
        cached_count = self._last_old_turn_count.get(session_id, 0)
        if old_count != cached_count or session_id not in self._cache:
            self._cache[session_id] = self._compress_turns(old_turns)
            self._last_old_turn_count[session_id] = old_count

        summary = self._cache.get(session_id, "")
        parts: List[str] = []

        if summary:
            parts.append("【历史对话摘要】" + summary)
        parts.append("【最近对话】")
        parts.append(self._format_recent(recent_turns, current_query))

        return "\n".join(parts)

    def reset_session(self, session_id: str) -> None:
        """清除某个 session 的压缩缓存。"""
        self._cache.pop(session_id, None)
        self._last_old_turn_count.pop(session_id, None)

    # ——— 内部方法 ———

    def _format_recent(
        self, turns: List[Dict[str, Any]], current_query: str
    ) -> str:
        """格式化最近轮次为对话文本。"""
        lines = []
        for msg in turns:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"用户：{content}")
            elif role == "assistant":
                # 只取前 300 字，避免旧回答过长
                short = content[:300]
                if len(content) > 300:
                    short += "……"
                lines.append(f"助手：{short}")
        lines.append(f"用户：{current_query}")
        return "\n".join(lines)

    def _compress_turns(self, turns: List[Dict[str, Any]]) -> str:
        """用 LLM 压缩旧对话轮次为一句话摘要。"""
        if not turns:
            return ""

        # 只取 user/assistant 的 content，限制长度
        dialogue_parts = []
        for msg in turns:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = (msg.get("content") or "")[:200]
            dialogue_parts.append(f"{role}：{content}")

        dialogue = "\n".join(dialogue_parts)
        prompt = f"""请用一段话（不超过 100 字）总结以下历史对话中用户问了哪些问题。

{dialogue}

总结："""

        try:
            response = self.llm.generate(
                system_prompt="你是对话摘要助手，只输出一句简洁的摘要。",
                user_prompt=prompt,
                temperature=0.1,
                max_tokens=150,
                thinking=False,
                stage="conversation_compress",
                timeout=5.0,
            )
            return (response or "").strip()
        except Exception:
            # 压缩失败时返回简单摘要，不阻塞主流程
            return ""
