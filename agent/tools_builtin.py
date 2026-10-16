"""内置工具：计算器（AST 白名单求值）与当前时间。

两个工具都在讲同一件事——**把模型的输出当作不可信输入**：
- 计算器绝不用 `eval`：用 AST 白名单只允许四则运算与少量数学函数，屏蔽属性访问/导入/下标等一切副作用入口；
- 时间工具不接受任意字符串格式，只接受受控的时区名，避免注入与非法输入。
"""

import ast
import json
import math
import operator
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import BaseTool, tool

MAX_EXPRESSION_LEN = 200

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_FUNCS: Dict[str, Any] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval(expression: str) -> float | int:
    """只允许数字、四则运算、白名单函数的表达式求值（无 eval / 无属性访问 / 无导入）。"""
    expression = (expression or "").strip()
    if not expression:
        raise ValueError("表达式为空")
    if len(expression) > MAX_EXPRESSION_LEN:
        raise ValueError(f"表达式过长（上限 {MAX_EXPRESSION_LEN} 字符）")

    tree = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            # 注意：不要把 int 一律转成 float —— round(2.345, 2) 的第二个参数必须是 int
            if isinstance(node.value, bool):
                raise ValueError("不支持布尔值")
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"不支持的常量类型：{type(node.value).__name__}")
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的运算符：{type(node.op).__name__}")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的一元运算符：{type(node.op).__name__}")
            return op(_eval(node.operand))
        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_CONSTS:
                return _ALLOWED_CONSTS[node.id]
            raise ValueError(f"不支持的变量名：{node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise ValueError("只允许调用白名单函数：" + ", ".join(sorted(_ALLOWED_FUNCS)))
            if node.keywords:
                raise ValueError("不支持关键字参数")
            args = [_eval(arg) for arg in node.args]
            result = _ALLOWED_FUNCS[node.func.id](*args)
            if not isinstance(result, (int, float)):
                raise ValueError("函数返回值必须是数字")
            return result
        raise ValueError(f"不支持的语法节点：{type(node).__name__}")

    value = _eval(tree)
    return value if isinstance(value, int) else float(value)


@tool
def calculator(expression: str) -> str:
    """数学计算器。输入一个数学表达式（如 "(3+5)*2"、"sqrt(16)+2**10"），返回计算结果。

    只支持数字、+ - * / // % ** 与白名单函数（sqrt/log/log10/exp/abs/round/min/max/pow/sin/cos/tan）。
    需要精确运算时用它，不要心算。
    """
    try:
        value = safe_eval(expression)
    except ZeroDivisionError:
        return json.dumps(
            {"ok": False, "expression": expression, "error": "除数为 0：请检查表达式"},
            ensure_ascii=False,
        )
    except Exception as exc:  # noqa: BLE001 —— 参数错误要变成可读反馈，而不是让流程崩掉
        return json.dumps(
            {
                "ok": False,
                "expression": expression,
                "error": f"{type(exc).__name__}: {exc}",
                "hint": "只支持数字与 + - * / // % ** 及白名单函数（sqrt/log/exp/abs/round/min/max/pow/sin/cos/tan）",
            },
            ensure_ascii=False,
        )

    if value == int(value) and abs(value) < 1e15:
        value = int(value)  # 整数结果就不要显示成 20.0
    return json.dumps({"ok": True, "expression": expression, "result": value}, ensure_ascii=False)


def get_now(tz_name: str = "Asia/Shanghai") -> datetime:
    """按时区名取当前时间；Windows 上没有 tzdata 时回退到固定 UTC+8。"""
    try:
        return datetime.now(ZoneInfo(tz_name))
    except (ZoneInfoNotFoundError, KeyError, ValueError):
        if tz_name in ("Asia/Shanghai", "Asia/Chongqing", "PRC"):
            return datetime.now(timezone(timedelta(hours=8)))
        raise ValueError(f"未知时区：{tz_name}")


def make_time_tool(tz_name: str = "Asia/Shanghai") -> BaseTool:
    """按配置生成时间工具（时区在构造时固定，模型无法传入任意时区字符串）。"""

    @tool("current_time")
    def current_time() -> str:
        """获取当前日期与时间（东八区）。需要「今天/现在」这类信息时调用。

        重要：任何涉及「今天」「现在」「今年」的问题都要先调用它拿真实时间，不要用模型记忆里的日期。
        """
        now = get_now(tz_name)
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        return json.dumps(
            {
                "timezone": tz_name,
                "iso": now.isoformat(timespec="seconds"),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "weekday": weekdays[now.weekday()],
                "human": now.strftime("%Y年%m月%d日 %H:%M:%S") + " " + weekdays[now.weekday()],
            },
            ensure_ascii=False,
        )

    return current_time


def build_builtin_tools(config: Dict[str, Any]) -> Dict[str, BaseTool]:
    """构建内置工具集（计算器 + 时间）。"""
    tz_name = str(config["tools"].get("timezone", "Asia/Shanghai"))
    tools = [calculator, make_time_tool(tz_name)]
    return {t.name: t for t in tools}
