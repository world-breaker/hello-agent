"""计算器工具的测试：重点是安全边界（把模型输出当不可信输入）。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.tools_builtin import calculator, get_now, safe_eval  # noqa: E402


def test_basic_arithmetic():
    assert safe_eval("(3+5)*2") == 16
    assert safe_eval("2**10") == 1024
    assert safe_eval("10/4") == 2.5
    assert safe_eval("7//2") == 3
    assert safe_eval("2*pi") == pytest.approx(6.28318, rel=1e-4)


def test_whitelisted_functions():
    assert safe_eval("sqrt(16) + abs(-3)") == 7
    assert safe_eval("round(2.345, 2)") == 2.35
    assert safe_eval("max(1, 5, 3)") == 5


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",  # 导入
        "open('secret.txt').read()",  # 文件访问
        "(1).__class__.__mro__",  # 属性穿透
        "lambda x: x",  # lambda
        "[1,2,3][0]",  # 下标
        "{'a':1}",  # 字典
        "eval('1+1')",  # 嵌套 eval
        "'abc' * 3",  # 字符串运算
    ],
)
def test_dangerous_expressions_rejected(expression):
    with pytest.raises(ValueError):
        safe_eval(expression)


def test_length_limit():
    with pytest.raises(ValueError, match="过长"):
        safe_eval("1+" * 200)


def test_tool_returns_json_and_never_raises():
    ok = calculator.invoke({"expression": "(1+2)*3"})
    assert '"result": 9' in ok

    division_by_zero = calculator.invoke({"expression": "1/0"})
    assert "除数为 0" in division_by_zero

    bad_syntax = calculator.invoke({"expression": "__import__('os')"})
    assert '"ok": false' in bad_syntax
    assert "白名单" in bad_syntax


def test_time_tool_shape():
    from agent.tools_builtin import make_time_tool

    tool = make_time_tool("Asia/Shanghai")
    payload = tool.invoke({})
    assert '"timezone": "Asia/Shanghai"' in payload
    assert '"weekday"' in payload
    # 时区名不受模型控制（构造时固定），异常时区应报错而不是静默返回错误时间
    now = get_now("Asia/Shanghai")
    assert now.utcoffset().total_seconds() == 8 * 3600
