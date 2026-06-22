import importlib
import sys

from repair_top_level_imports import rewrite_generated_inits_to_lazy


def test_rewrite_generated_inits_to_lazy_preserves_top_package_import(tmp_path, monkeypatch):
    generated_root = tmp_path / "generated_code"
    package_dir = generated_root / "pkg"
    package_dir.mkdir(parents=True)
    (package_dir / "ok.py").write_text("class Exported:\n    pass\n", encoding="utf-8")
    (package_dir / "bad.py").write_text('raise RuntimeError("bad eager import")\n', encoding="utf-8")
    (package_dir / "__init__.py").write_text(
        "\n".join(
            [
                "# AUTO-GENERATED PACKAGE EXPORTS",
                "# This file is generated to provide stable package/subpackage imports.",
                "",
                "from .ok import Exported",
                "from . import bad",
                "",
                "__all__ = [",
                '    "Exported",',
                '    "bad",',
                "]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rewritten = rewrite_generated_inits_to_lazy(generated_root)

    assert rewritten == [str(package_dir / "__init__.py")]
    init_text = (package_dir / "__init__.py").read_text(encoding="utf-8")
    assert "def __getattr__(name):" in init_text
    assert "from . import bad" not in init_text

    monkeypatch.syspath_prepend(str(generated_root))
    sys.modules.pop("pkg", None)
    sys.modules.pop("pkg.bad", None)
    pkg = importlib.import_module("pkg")

    assert pkg.Exported.__name__ == "Exported"
    try:
        getattr(pkg, "bad")
    except RuntimeError as exc:
        assert "bad eager import" in str(exc)
    else:
        raise AssertionError("lazy bad module import did not raise")
