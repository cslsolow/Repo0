import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "regenerate_dependency_graph.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("regenerate_dependency_graph", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_artifact_paths_points_to_expected_files():
    module = _load_script_module()
    agents_output = Path("/tmp/example/agents_output")

    paths = module.resolve_artifact_paths(agents_output)

    assert paths["memory"] == agents_output / "memory.json"
    assert paths["requirement_dag"] == agents_output / "requirement_dag.json"
    assert paths["dependency_graph"] == agents_output / "dependency_graph.json"


def test_resolve_output_path_uses_override_when_provided():
    module = _load_script_module()
    agents_output = Path("/tmp/example/agents_output")
    override = Path("/tmp/example/custom/dependency_graph.preview.json")

    resolved = module.resolve_output_path(agents_output, override)

    assert resolved == override


def test_load_dependency_inputs_prefers_cached_dependency_graph(tmp_path: Path):
    module = _load_script_module()
    agents_output = tmp_path / "agents_output"
    agents_output.mkdir()

    dependency_graph_path = agents_output / "dependency_graph.json"
    dependency_graph_path.write_text(
        json.dumps(
            {
                "components": [{"id": "ReqA::Alpha"}],
                "requirement_edges": [{"source": "ReqA", "target": "ReqB"}],
                "dependency_graph": {},
            }
        ),
        encoding="utf-8",
    )

    paths = module.resolve_artifact_paths(agents_output)
    loaded = module.load_dependency_inputs(paths, workspace_root=tmp_path)

    assert loaded["source"] == "cached_dependency_graph"
    assert loaded["components"] == [{"id": "ReqA::Alpha"}]
    assert loaded["requirement_edges"] == [{"source": "ReqA", "target": "ReqB"}]


def test_build_run_summary_reports_counts_and_paths():
    module = _load_script_module()

    summary = module.build_run_summary(
        loaded_inputs={
            "source": "cached_dependency_graph",
            "components": [{"id": "ReqA::Alpha"}, {"id": "ReqB::Beta"}],
            "requirement_edges": [{"source": "ReqA", "target": "ReqB"}],
        },
        output_path=Path("/tmp/out.json"),
        dependency_graph={
            "component_edges": [{"source": "ReqA::Alpha", "target": "ReqB::Beta"}],
            "edges": [{"source": "ReqA", "target": "ReqB"}],
            "uncertain_edges": [{"source_hint": "ReqA::Alpha", "target_hint": "ReqC::Gamma"}],
        },
    )

    assert summary["input_source"] == "cached_dependency_graph"
    assert summary["components"] == 2
    assert summary["input_requirement_edges"] == 1
    assert summary["component_edges"] == 1
    assert summary["requirement_edges"] == 1
    assert summary["uncertain_edges"] == 1
    assert summary["dependency_graph_path"] == "/tmp/out.json"
