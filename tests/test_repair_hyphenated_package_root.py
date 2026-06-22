import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "repair_hyphenated_package_root.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("repair_hyphenated_package_root", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_repair_tree_normalizes_hyphenated_package_paths_and_imports(tmp_path: Path) -> None:
    source = tmp_path / "generated_code" / "my-repo" / "pkg-one"
    source.mkdir(parents=True)
    py_file = source / "mod-file.py"
    py_file.write_text(
        "from my-repo.pkg-one.mod-file import Thing\n"
        "import my-repo.pkg-one.mod-file\n",
        encoding="utf-8",
    )

    repair = _load_script()
    report = repair.repair_tree(tmp_path / "generated_code")

    fixed = tmp_path / "generated_code" / "my_repo" / "pkg_one" / "mod_file.py"
    assert fixed.exists()
    assert not py_file.exists()
    assert fixed.read_text(encoding="utf-8") == (
        "from my_repo.pkg_one.mod_file import Thing\n"
        "import my_repo.pkg_one.mod_file\n"
    )
    assert report["renamed"] == 3
    assert report["rewritten"] == 1
