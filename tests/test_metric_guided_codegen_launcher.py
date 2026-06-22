import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "metric_guided_codegen_launcher.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("metric_guided_codegen_launcher", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_baseline_repo(root: Path, repo: str) -> Path:
    repo_dir = root / repo
    agents_output = repo_dir / "agents_output"
    readme_output = repo_dir / "readme_output"
    agents_output.mkdir(parents=True)
    readme_output.mkdir(parents=True)
    (agents_output / "architectures.json").write_text(json.dumps([{"task": "ReqA"}]), encoding="utf-8")
    (agents_output / "decomposed_dag.json").write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")
    (agents_output / "requirements_merge_result.json").write_text(json.dumps({"requirements_after_merge": []}), encoding="utf-8")
    (agents_output / "requirements_for_dag.json").write_text(json.dumps({"requirements": []}), encoding="utf-8")
    (agents_output / "edges_for_dag.json").write_text(json.dumps({"edges": []}), encoding="utf-8")
    (agents_output / "requirement_dag.json").write_text(json.dumps({"nodes": {}, "adjacency": {}}), encoding="utf-8")
    (agents_output / "plan.json").write_text(json.dumps({"plan": []}), encoding="utf-8")
    (agents_output / "component_merge_report.json").write_text(json.dumps({"stats": {}}), encoding="utf-8")
    (agents_output / "dependency_graph.json").write_text(json.dumps({"dependency_graph": {"edges": []}}), encoding="utf-8")
    (agents_output / "layout_grouping_report.json").write_text(json.dumps({"components": []}), encoding="utf-8")
    (agents_output / "package_api_plan.json").write_text(json.dumps({"components": []}), encoding="utf-8")
    (agents_output / "actions.json").write_text(json.dumps([{"task": "ReqA", "actions": []}]), encoding="utf-8")
    (agents_output / "action_refinement_report.json").write_text(json.dumps({"stats": {"components_after": 1}}), encoding="utf-8")
    (agents_output / "component_metric_action_report.json").write_text(json.dumps({"stats": {}}), encoding="utf-8")
    (agents_output / "gap_addition_report.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
    (readme_output / "requirements.json").write_text(json.dumps([{"name": "ReqA"}]), encoding="utf-8")
    return repo_dir


def _build_config(module, tmp_path: Path, repo: str = "requests"):
    workspace_dir = tmp_path / "workspace"
    baseline_root = tmp_path / "baseline"
    output_root = tmp_path / "runs"
    readme_req_source = tmp_path / "reqsrc" / repo / "README.req"
    requirements_source = tmp_path / "reqsrc" / repo / "requirements.json"
    workspace_dir.mkdir()
    readme_req_source.parent.mkdir(parents=True)
    readme_req_source.write_text("dummy req\n", encoding="utf-8")
    requirements_source.write_text(
        json.dumps({"project_summary": "from-source", "requirements": [{"name": "ReqNew"}]}),
        encoding="utf-8",
    )
    _make_baseline_repo(baseline_root, repo)
    return module.LauncherConfig(
        repo=repo,
        workspace_dir=workspace_dir,
        baseline_root=baseline_root,
        output_root=output_root,
        run_label="metric_guided_codegen",
        run_ts="20260605_123000",
        python_bin=Path("/usr/bin/python3"),
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model="gpt-5-mini",
        reasoning_effort="medium",
        max_workers=4,
        postcheck_max_workers=2,
        log_level="INFO",
        parent_codegen_dag_source="dependency",
        action_refinement_rounds=5,
        action_refinement_stop_on_stable=True,
        action_refinement_save_stops_component=False,
        use_processes=True,
        enable_component_metric_actions=True,
        enable_component_metric_merge_judge=True,
        component_metric_split_cohesion_threshold=2.0 / 3.0,
        component_metric_split_min_subrequirements=3,
        component_split_min_confidence=0.7,
        component_metric_merge_max_small_subrequirements=1,
        retry_empty_generated_components=True,
        readme_req_source=readme_req_source,
        requirements_source=requirements_source,
        enable_gap_add_actions=True,
        gap_add_proposal_threshold=0.55,
        gap_add_component_threshold=0.74,
        gap_add_requirement_threshold=0.82,
        stop_after_architecture_refinement=True,
    )


def test_prepare_launcher_run_copies_baseline_and_removes_reused_artifacts(tmp_path: Path):
    module = _load_script_module()
    config = _build_config(module, tmp_path)

    summary = module.prepare_launcher_run(config)

    run_dir = Path(summary["run_dir"])
    repo_root = Path(summary["repo_root"])
    output_dir = Path(summary["output_dir"])
    assert run_dir.exists()
    assert repo_root.exists()
    assert (repo_root / "readme_output" / "requirements.json").exists()
    assert (repo_root / "README.req").read_text(encoding="utf-8") == "dummy req\n"
    req_payload = json.loads((repo_root / "readme_output" / "requirements.json").read_text(encoding="utf-8"))
    assert req_payload["project_summary"] == "from-source"
    assert req_payload["requirements"][0]["name"] == "ReqNew"
    assert not (output_dir / "architectures.json").exists()
    assert not (output_dir / "decomposed_dag.json").exists()
    assert not (output_dir / "requirements_merge_result.json").exists()
    assert not (output_dir / "requirements_for_dag.json").exists()
    assert not (output_dir / "edges_for_dag.json").exists()
    assert not (output_dir / "requirement_dag.json").exists()
    assert not (output_dir / "plan.json").exists()
    assert not (output_dir / "dependency_graph.json").exists()
    assert not (output_dir / "layout_grouping_report.json").exists()
    assert not (output_dir / "package_api_plan.json").exists()
    assert not (output_dir / "actions.json").exists()
    assert not (output_dir / "action_refinement_report.json").exists()
    assert not (output_dir / "component_metric_action_report.json").exists()
    assert not (output_dir / "gap_addition_report.json").exists()
    assert Path(summary["launch_script"]).exists()
    assert Path(summary["manifest_file"]).exists()


def test_build_run_agents_command_includes_metric_guided_flags(tmp_path: Path):
    module = _load_script_module()
    config = _build_config(module, tmp_path, repo="statsmodels")
    paths = module.resolve_run_paths(config)

    command = module.build_run_agents_command(config, paths)

    assert "--enable-component-metric-actions" in command
    assert "--enable-component-metric-merge-judge" in command
    assert "--enable-gap-add-actions" in command
    assert "--stop-after-architecture-refinement" in command
    assert "--gap-add-proposal-threshold" in command
    assert "0.55" in command
    assert "--gap-add-component-threshold" in command
    assert "0.74" in command
    assert "--gap-add-requirement-threshold" in command
    assert "0.82" in command
    assert "--action-refinement-rounds" in command
    assert "5" in command
    assert "--action-refinement-stop-on-stable" in command
    assert "--component-split-min-confidence" in command
    assert "0.7" in command
    assert "--repo" in command
    assert "statsmodels" in command
    assert "--output" in command
    assert str(paths.output_dir) in command


def test_write_manifest_redacts_api_key(tmp_path: Path):
    module = _load_script_module()
    config = _build_config(module, tmp_path)
    paths = module.resolve_run_paths(config)
    paths.run_dir.mkdir(parents=True)

    command = module.build_run_agents_command(config, paths)
    module.write_manifest(config, paths, command)

    manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    joined = " ".join(manifest["command"])
    assert "test-key" not in joined
    assert "<redacted>" in joined
