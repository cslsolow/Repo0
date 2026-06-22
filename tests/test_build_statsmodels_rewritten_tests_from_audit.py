import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_statsmodels_rewritten_tests_from_audit.py"

spec = importlib.util.spec_from_file_location("build_statsmodels_rewritten_tests_from_audit", SCRIPT)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_write_suite_keeps_only_passing_functions(tmp_path):
    golden_repo = tmp_path / "golden"
    test_file = golden_repo / "pkg" / "tests" / "test_demo.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        '''import pytest

VALUE = 1

def helper():
    return 3

def test_ok():
    assert helper() == 3

def test_bad():
    assert False

class TestGroup:
    FLAG = True

    def test_drop_first(self):
        assert False

    def test_keep(self):
        assert self.FLAG

    def test_drop(self):
        assert False
''',
        encoding="utf-8",
    )

    audit_entries = {
        "pkg/tests/test_demo.py": [
            mod.TaskEntry("t1", "pkg/tests/test_demo.py", "test_demo", "test_ok", "passed"),
            mod.TaskEntry("t2", "pkg/tests/test_demo.py", "test_demo", "test_bad", "failed"),
            mod.TaskEntry("t3", "pkg/tests/test_demo.py", "class TestGroup", "test_keep", "passed"),
            mod.TaskEntry("t4", "pkg/tests/test_demo.py", "class TestGroup", "test_drop", "failed"),
        ]
    }

    manifest = mod.write_suite(
        golden_repo=golden_repo,
        audit_entries=audit_entries,
        output_root=tmp_path / "out",
        min_pass_ratio=0.5,
    )

    rewritten = (tmp_path / "out" / "rewritten_tests" / "pkg" / "tests" / "test_demo.py").read_text(encoding="utf-8")
    assert "def test_ok" in rewritten
    assert "def test_bad" not in rewritten
    assert "def test_keep" in rewritten
    assert "def test_drop_first" not in rewritten
    assert "def test_drop" not in rewritten
    assert manifest["selected_file_count"] == 1
    assert manifest["selected_task_count"] == 2


def test_should_keep_file_uses_pass_ratio():
    entries = [
        mod.TaskEntry("t1", "f.py", "m", "a", "passed"),
        mod.TaskEntry("t2", "f.py", "m", "b", "failed"),
        mod.TaskEntry("t3", "f.py", "m", "c", "skipped"),
    ]

    assert mod.should_keep_file(entries, 0.66) is True
    assert mod.should_keep_file(entries, 0.67) is False


def test_write_suite_persists_manifest(tmp_path):
    golden_repo = tmp_path / "golden"
    source = golden_repo / "pkg" / "tests" / "test_one.py"
    source.parent.mkdir(parents=True)
    source.write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    audit_entries = {
        "pkg/tests/test_one.py": [
            mod.TaskEntry("t1", "pkg/tests/test_one.py", "test_one", "test_ok", "passed"),
        ]
    }
    mod.write_suite(
        golden_repo=golden_repo,
        audit_entries=audit_entries,
        output_root=tmp_path / "out",
        min_pass_ratio=1.0,
    )

    manifest_path = tmp_path / "out" / "rewritten_tests_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["selected_file_count"] == 1
    assert manifest["selected_files"][0]["file"] == "pkg/tests/test_one.py"
