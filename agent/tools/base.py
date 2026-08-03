"""
Agent 工具基类与内置工具。

Agent 的强大之处在于可以调用外部工具扩展能力。
每个工具是一个独立的可调用单元，有明确的输入输出规范。

核心接口：
    tool = CalculatorTool()
    result = tool.execute(expression="2 + 2")
    # 返回: ToolResult(output=4, success=True)
"""

import ast
import hashlib
import json
import logging
import operator
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# 安全数学表达式解析器（替代 eval）
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def _safe_eval(expr: str) -> float:
    """
    安全地计算数学表达式，仅允许数字和基本运算符。
    完全避免 eval() 的安全风险。
    """
    tree = ast.parse(expr, mode='eval')

    def _eval(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("表达式中只允许数字")
        elif isinstance(node, ast.BinOp):
            op = _SAFE_OPS.get(type(node.op))
            if not op:
                raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
            return op(_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            op = _SAFE_OPS.get(type(node.op))
            if not op:
                raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
            return op(_eval(node.operand))
        elif isinstance(node, ast.Expression):
            return _eval(node.body)
        else:
            raise ValueError(f"不支持的表达式类型: {type(node).__name__}")

    return _eval(tree)


@dataclass
class ToolResult:
    """工具执行结果"""
    tool_name: str
    input_params: Dict[str, Any]
    output: Any
    success: bool
    error_message: str = ""


class ToolCallGuard:
    """断路器：防止死循环、重复调用和超时。"""

    def __init__(self, max_rounds: int = 5, max_total_seconds: float = 10.0):
        self.max_rounds = max_rounds
        self.max_total_seconds = max_total_seconds
        self.call_history: List[str] = []  # 存调用指纹
        self.rounds = 0
        self.started = time.monotonic()

    def check(self, tool_name: str, params: dict) -> None:
        """每次工具调用前检查。抛出异常则终止。"""
        self.rounds += 1

        if self.rounds > self.max_rounds:
            raise RuntimeError(
                f"工具调用已达上限 {self.max_rounds} 轮，终止避免死循环"
            )

        if time.monotonic() - self.started > self.max_total_seconds:
            raise TimeoutError(
                f"工具调用总耗时超过 {self.max_total_seconds} 秒"
            )

        # 重复调用检测：相同的工具名 + 相同的参数 = 死循环
        fingerprint = hashlib.md5(
            json.dumps({"tool": tool_name, "params": params}, sort_keys=True).encode()
        ).hexdigest()
        if fingerprint in self.call_history:
            raise RuntimeError(
                f"检测到重复调用 {tool_name}({params})，终止避免死循环"
        )
        self.call_history.append(fingerprint)


class BaseTool(ABC):
    """
    工具抽象基类

    每个工具必须定义：
    - name: 工具名称（Agent 通过名称调用）
    - description: 功能描述（LLM 通过描述理解何时使用该工具）
    - input_schema: 输入参数规范（JSON Schema 格式）
    - execute: 执行逻辑
    """

    name: str = ""
    description: str = ""
    input_schema: Dict = {}

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass

    def validate_params(self, **kwargs) -> Optional[str]:
        """校验参数是否符合 input_schema。返回 None 表示通过，否则返回错误描述。"""
        schema = self.input_schema or {}
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for param in required:
            if param not in kwargs:
                return f"缺少必填参数 '{param}'"
        for param in properties:
            if param in kwargs:
                prop_schema = properties[param]
                pattern = prop_schema.get("pattern")
                if pattern and not re.search(pattern, str(kwargs[param])):
                    return f"参数 '{param}' 的值 '{kwargs[param]}' 不匹配格式要求"
                max_len = prop_schema.get("maxLength")
                if max_len and len(str(kwargs[param])) > max_len:
                    return f"参数 '{param}' 超过最大长度 {max_len}"
        return None

    def get_tool_info(self) -> Dict[str, Any]:
        """返回工具信息，用于 LLM 的工具选择决策"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }


class CalculatorTool(BaseTool):
    """
    计算器工具

    用于执行数学运算。LLM 在处理涉及数字的问题时，
    直接推理容易出错（如大数乘法），调用计算器可以保证准确性。
    """

    name = "calculator"
    description = (
        "执行数学运算，支持加减乘除、括号和百分比。"
        "参数 expression 为数学表达式字符串。"
        "示例：TOOL:calculator\\nARGS:{\"expression\":\"320*5000\"}"
        "注意：只支持 + - * / ( ) % 运算，不支持函数调用"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，如 '123 * 456' 或 '(100 - 20) / 5'",
                "pattern": r"^[\d+\-*/.()%\s]+$",
                "maxLength": 200,
            }
        },
        "required": ["expression"]
    }

    def execute(self, expression: str) -> ToolResult:
        try:
            expression = str(expression).strip()
            if not expression:
                raise ValueError("表达式不能为空")
            if len(expression) > 200:
                raise ValueError(f"表达式过长（{len(expression)} 字符，上限 200）")

            allowed_chars = set("0123456789+-*/.() %")
            if not all(c in allowed_chars for c in expression.replace(" ", "")):
                raise ValueError("表达式包含非法字符，只支持数字和 +-*/.()%")

            result = _safe_eval(expression)
            return ToolResult(
                tool_name=self.name,
                input_params={"expression": expression},
                output=result,
                success=True
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                input_params={"expression": expression},
                output=None,
                success=False,
                error_message=str(e)
            )


class DatabaseQueryTool(BaseTool):
    """
    数据库查询工具（示例）

    用于执行结构化数据查询。实际项目中可连接 MySQL/PostgreSQL/ClickHouse 等。
    这里用模拟数据演示。
    """

    name = "database_query"
    description = (
        "查询结构化数据库中的销售数据。"
        "参数 sql 为 SELECT 查询语句。"
        "示例：TOOL:database_query\\nARGS:{\"sql\":\"SELECT * FROM sales\"}"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "SQL 查询语句（仅支持 SELECT）",
                "maxLength": 1000,
            }
        },
        "required": ["sql"]
    }

    def __init__(self):
        self.mock_data = {
            "sales": [
                {"product": "产品A", "month": "2024-01", "amount": 150000},
                {"product": "产品B", "month": "2024-01", "amount": 230000},
                {"product": "产品C", "month": "2024-01", "amount": 89000},
                {"product": "产品A", "month": "2024-02", "amount": 180000},
                {"product": "产品B", "month": "2024-02", "amount": 210000},
            ]
        }

    def execute(self, sql: str) -> ToolResult:
        try:
            sql = str(sql).strip()
            if not sql:
                raise ValueError("SQL 不能为空")
            sql_lower = sql.lower()
            if not sql_lower.startswith("select"):
                raise ValueError("仅支持 SELECT 查询")
            if "drop" in sql_lower or "delete" in sql_lower or "insert" in sql_lower:
                raise ValueError("不允许修改数据的 SQL 语句")

            if "from sales" in sql_lower:
                results = self.mock_data["sales"]
                return ToolResult(
                    tool_name=self.name,
                    input_params={"sql": sql},
                    output=results,
                    success=True
                )
            else:
                return ToolResult(
                    tool_name=self.name,
                    input_params={"sql": sql},
                    output=[],
                    success=True
                )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                input_params={"sql": sql},
                output=None,
                success=False,
                error_message=str(e)
            )


class ToolRegistry:
    """
    工具注册表

    管理所有可用工具，Agent 通过名称查找和调用。
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """注册工具"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """获取工具（精确匹配）"""
        if name not in self._tools:
            raise ValueError(f"未知工具: {name}")
        return self._tools[name]

    def fuzzy_find(self, raw_name: str) -> str:
        """模糊匹配工具名，防 LLM 拼写错误。返回标准化名称。"""
        normalized = raw_name.lower().strip()

        # 精确匹配
        if normalized in self._tools:
            return normalized

        # 子串匹配
        for name in self._tools:
            if normalized in name or name in normalized:
                logger.warning("工具名模糊匹配：%s → %s", raw_name, name)
                return name

        raise ValueError(
            f"未知工具 '{raw_name}'，可用：{list(self._tools)}"
        )

    def list_tools(self) -> list:
        """列出所有工具信息"""
        return [tool.get_tool_info() for tool in self._tools.values()]

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """执行指定工具（含参数校验）"""
        tool = self.get(tool_name)
        validation_error = tool.validate_params(**kwargs)
        if validation_error:
            return ToolResult(
                tool_name=tool_name,
                input_params=kwargs,
                output=None,
                success=False,
                error_message=f"参数校验失败：{validation_error}"
            )
        return tool.execute(**kwargs)


# 预置工具集
DEFAULT_TOOLS = ToolRegistry()
DEFAULT_TOOLS.register(CalculatorTool())
DEFAULT_TOOLS.register(DatabaseQueryTool())
