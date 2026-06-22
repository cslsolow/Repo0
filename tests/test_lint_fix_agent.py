from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.coding.lint_fix_agent import LintFixAgent  # noqa: E402
from agents.coding.static_preflight import run_static_preflight  # noqa: E402


def test_lint_fix_agent_categorizes_diagnostics():
    text = "SyntaxError: invalid syntax\n\nF401 imported but unused\n\ncannot resolve 'x' from 'pkg.mod' (static)"
    categories = LintFixAgent._categorize_diagnostics(text)

    assert "syntax" in categories
    assert "ruff_unused_import" in categories
    assert "missing_symbol" in categories


def test_lint_fix_agent_collects_all_generated_python_files(tmp_path: Path):
    root = tmp_path / "generated_code"
    root.mkdir()
    entry_file = root / "entry_file.py"
    discovered_file = root / "discovered_file.py"
    non_python = root / "notes.txt"
    entry_file.write_text("x = 1\n", encoding="utf-8")
    discovered_file.write_text("y = 2\n", encoding="utf-8")
    non_python.write_text("not python\n", encoding="utf-8")

    agent = LintFixAgent(api_config={"enable_lint_llm_fix": False})
    collected = agent._collect_python_files(root, [{"files": {"only": str(entry_file)}}])

    assert collected == sorted([entry_file.resolve(), discovered_file.resolve()], key=lambda p: str(p))


def test_static_preflight_reports_issue_kind_counts(tmp_path: Path):
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    bad = root / "consumer.py"
    bad.write_text("from .missing import thing\n", encoding="utf-8")

    report = run_static_preflight(tmp_path, [bad])

    assert report["issue_count"] >= 1
    assert report["issue_kind_counts"].get("import_path", 0) >= 1
