import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIX_AGENT_PATH = ROOT / "agents" / "coding" / "fix_agent.py"

spec = importlib.util.spec_from_file_location("fix_agent", FIX_AGENT_PATH)
assert spec is not None and spec.loader is not None
fix_agent_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fix_agent_mod)
FixAgent = fix_agent_mod.FixAgent


def _print_diff(title: str, before: str, after: str) -> None:
    print(f"\n===== {title} =====")
    print("----- BEFORE -----")
    print(before)
    print("----- AFTER ------")
    print(after)


def test_fix_python_content_handles_unterminated_single_line_string_in_function():
    agent = FixAgent()
    source = 'def run():\n    "missing close\n    return 1\n'

    result = agent.fix_python_content(source)
    _print_diff("unterminated single-line string", source, result["fixed_content"])

    assert result["fixed"] is True
    fixed = result["fixed_content"]
    compile(fixed, "<fixed>", "exec")
    assert '"missing close"' in fixed


def test_fix_python_content_handles_unterminated_triple_quote_in_function():
    agent = FixAgent()
    source = 'def run():\n    """missing close\n    return 1\n'

    result = agent.fix_python_content(source)
    _print_diff("unterminated triple quote", source, result["fixed_content"])

    assert result["fixed"] is True
    fixed = result["fixed_content"]
    compile(fixed, "<fixed>", "exec")
    assert fixed.count('"""') % 2 == 0


def test_fix_python_content_adds_missing_colon():
    agent = FixAgent()
    source = "def run(x)\n    return x\n"

    result = agent.fix_python_content(source)
    _print_diff("missing colon", source, result["fixed_content"])

    assert result["fixed"] is True
    fixed = result["fixed_content"]
    compile(fixed, "<fixed>", "exec")
    assert "def run(x):" in fixed


def test_fix_files_prefers_compile_error_target_path():
    agent = FixAgent()
    related_files = {
        "pkg/a.py": 'def run():\n    """oops\n    return 1\n',
        "pkg/b.py": "def ok():\n    return 1\n",
    }
    compile_error = (
        'Traceback ...\n  File "/tmp/project/pkg/a.py", line 2\n'
        '    """oops\n'
        "SyntaxError: unterminated triple-quoted string literal (detected at line 3)"
    )

    result = agent.fix_files(related_files=related_files, compile_error=compile_error)
    _print_diff(
        "targeted file fix via compile_error",
        related_files["pkg/a.py"],
        result["updated_files"]["pkg/a.py"],
    )

    assert result["touched_files"] == ["pkg/a.py"]
    compile(result["updated_files"]["pkg/a.py"], "<fixed>", "exec")


def test_fix_python_content_handles_mid_file_module_indent_block():
    agent = FixAgent()
    source = (
        '"""module docs"""\n\n'
        'import os\n\n'
        '    from __future__ import annotations\n'
        '    import math\n\n'
        '    def plot_value():\n'
        '        return math.pi\n'
    )

    result = agent.fix_python_content(source)
    _print_diff("mid-file module indent block", source, result["fixed_content"])

    assert result["fixed"] is True
    fixed = result["fixed_content"]
    compile(fixed, "<fixed>", "exec")
    assert "\nfrom __future__ import annotations\n" in fixed
    assert "\ndef plot_value():\n" in fixed


def test_fix_python_content_closes_multiline_assert_parenthesis():
    agent = FixAgent()
    source = (
        "def test_x():\n"
        "    values = [1, 2]\n"
        "    assert (\n"
        "        sum(values) == 3\n"
    )

    result = agent.fix_python_content(source)
    _print_diff("multiline assert parenthesis", source, result["fixed_content"])

    assert result["fixed"] is True
    fixed = result["fixed_content"]
    compile(fixed, "<fixed>", "exec")
    assert "sum(values) == 3)" in fixed


def test_fix_python_content_removes_bare_star_in_call():
    agent = FixAgent()
    source = (
        "def run():\n"
        "    return metric(*, axis=1, keepdims=True)\n"
    )

    result = agent.fix_python_content(source)
    _print_diff("bare star in call", source, result["fixed_content"])

    assert result["fixed"] is True
    fixed = result["fixed_content"]
    compile(fixed, "<fixed>", "exec")
    assert "metric(axis=1, keepdims=True)" in fixed


def test_fix_python_content_converts_assignment_expression_to_comparison():
    agent = FixAgent()
    source = (
        "def run(x):\n"
        "    if (x = 1):\n"
        "        return True\n"
        "    return False\n"
    )

    result = agent.fix_python_content(source)
    _print_diff("assignment in expression", source, result["fixed_content"])

    assert result["fixed"] is True
    fixed = result["fixed_content"]
    compile(fixed, "<fixed>", "exec")
    assert "if (x == 1):" in fixed


def test_fix_python_content_removes_bare_star_before_kwargs_in_signature():
    agent = FixAgent()
    source = (
        "class Rotator:\n"
        "    def rotate(self, method: str, *, **kwargs) -> None:\n"
        "        return None\n"
    )

    result = agent.fix_python_content(source)
    _print_diff("bare star before kwargs in signature", source, result["fixed_content"])

    assert result["fixed"] is True
    fixed = result["fixed_content"]
    compile(fixed, "<fixed>", "exec")
    assert "def rotate(self, method: str, **kwargs) -> None:" in fixed


def test_fix_python_content_repairs_comma_after_function_name():
    agent = FixAgent()
    source = (
        "class Demo:\n"
        "    def likelihood_ratio,(self, *args, **kwargs) -> Any:\n"
        "        return None\n"
    )

    result = agent.fix_python_content(source)
    _print_diff("comma after function name", source, result["fixed_content"])

    assert result["fixed"] is True
    fixed = result["fixed_content"]
    compile("from typing import Any\n" + fixed, "<fixed>", "exec")
    assert "def likelihood_ratio(self, *args, **kwargs) -> Any:" in fixed


def test_fix_python_content_comments_out_bare_todo_line():
    agent = FixAgent()
    source = (
        "def run():\n"
        "    TODO implement this\n"
        "    return 1\n"
    )

    result = agent.fix_python_content(source)
    _print_diff("bare todo line", source, result["fixed_content"])

    assert result["fixed"] is True
    fixed = result["fixed_content"]
    compile(fixed, "<fixed>", "exec")
    assert "    # TODO implement this" in fixed


def test_fix_python_content_sanitizes_illegal_function_identifier():
    agent = FixAgent()
    source = (
        "class Demo:\n"
        "    def export/import(self, *args, **kwargs):\n"
        "        return None\n"
    )

    result = agent.fix_python_content(source)
    _print_diff("illegal function identifier", source, result["fixed_content"])

    assert result["fixed"] is True
    fixed = result["fixed_content"]
    compile(fixed, "<fixed>", "exec")
    assert "def export_import(self, *args, **kwargs):" in fixed
