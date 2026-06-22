import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("openai")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.dependency_graph_agent import DependencyGraphAgent  # noqa: E402
from run_agents import build_components_from_memory  # noqa: E402


def test_fallback_infer_component_dependencies_resolves_names_and_filters_self():
    components = [
        {
            "id": "ReqA::UserRepo",
            "name": "UserRepo",
            "requirement_node": "ReqA",
            "known_dependencies": ["DBAdapter", "ReqA::UserRepo"],
        },
        {
            "id": "ReqB::DBAdapter",
            "name": "DBAdapter",
            "requirement_node": "ReqB",
        },
    ]

    agent = DependencyGraphAgent(api_config={})
    result = agent.infer_component_dependencies(components)

    edges = result.get("cross_requirement_component_edges", [])
    assert len(edges) == 1
    assert edges[0]["source"] == "ReqA::UserRepo"
    assert edges[0]["target"] == "ReqB::DBAdapter"
    assert edges[0]["reason"] == "declared dependency"
    assert edges[0]["confidence"] == 0.6


def test_aggregate_requirement_dependencies_groups_and_skips_same_requirement():
    components = [
        {"id": "ReqA::Alpha", "name": "Alpha", "requirement_node": "ReqA"},
        {"id": "ReqB::Beta", "name": "Beta", "requirement_node": "ReqB"},
        {"id": "ReqB::Gamma", "name": "Gamma", "requirement_node": "ReqB"},
    ]
    edges = [
        {"source": "Alpha", "target": "ReqB::Beta", "confidence": 0.9, "reason": "calls Beta"},
        {"source": "ReqA::Alpha", "target": "Gamma", "confidence": 0.7, "reason": "calls Gamma"},
        {"source": "ReqB::Beta", "target": "Gamma", "confidence": 0.5, "reason": "same requirement"},
    ]

    agent = DependencyGraphAgent(api_config={})
    result = agent.aggregate_requirement_dependencies(components, edges)

    req_edges = result.get("edges", [])
    assert len(req_edges) == 1
    agg_edge = req_edges[0]
    assert agg_edge["source"] == "ReqA"
    assert agg_edge["target"] == "ReqB"
    assert agg_edge["count"] == 2
    assert agg_edge["confidence"] == pytest.approx(0.8, abs=1e-6)
    assert len(agg_edge["supporting_edges"]) == 2

    skipped = result.get("skipped_edges", [])
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "same requirement"


def test_build_components_from_memory_preserves_component_semantics():
    snapshot = SimpleNamespace(
        implemented_components={
            "ReqA::Alpha": SimpleNamespace(
                component_name="Alpha",
                requirement_node="ReqA",
                file_path="planned/alpha.py",
                exports=["AlphaAPI"],
                dependencies=[],
                class_names=["Alpha"],
                function_signatures=[],
                metadata={
                    "responsibilities": ["parse formulas", "validate terms"],
                    "serves_subrequirements": ["ReqA::parser"],
                    "parent_requirement": "ReqA",
                },
            )
        }
    )
    memory_agent = SimpleNamespace(snapshot=snapshot)

    components = build_components_from_memory(memory_agent)

    assert len(components) == 1
    assert components[0]["responsibilities"] == ["parse formulas", "validate terms"]
    assert components[0]["serves_subrequirements"] == ["ReqA::parser"]
    assert components[0]["parent_requirement"] == "ReqA"
    assert components[0]["exports"] == ["AlphaAPI"]


def test_build_requirement_dependency_edges_accepts_redesigned_llm_schema_without_pruning():
    components = [
        {
            "id": "ReqA::Alpha",
            "name": "Alpha",
            "requirement_node": "ReqA",
            "responsibilities": ["Provide canonical term descriptors"],
            "serves_subrequirements": ["ReqA::parser"],
        },
        {
            "id": "ReqB::Beta",
            "name": "Beta",
            "requirement_node": "ReqB",
            "responsibilities": ["Fit linear models using canonical descriptors"],
            "serves_subrequirements": ["ReqB::fit"],
        },
        {
            "id": "ReqB::Gamma",
            "name": "Gamma",
            "requirement_node": "ReqB",
            "responsibilities": ["Render reports"],
            "serves_subrequirements": ["ReqB::report"],
        },
    ]

    class _FakeLLM:
        def call_json(self, *_args, **_kwargs):
            return {
                "cross_requirement_edges": [
                    {
                        "source": "ReqB::Beta",
                        "target": "ReqA::Alpha",
                        "dependency_type": "api_call",
                        "reason": "Model fitting must consume canonical descriptors built by Alpha.",
                        "confidence": 0.93,
                        "evidence": ["Beta responsibilities mention canonical descriptors."],
                    }
                ],
                "same_requirement_edges": [
                    {
                        "source": "ReqB::Gamma",
                        "target": "ReqB::Beta",
                        "dependency_type": "shared_result_contract",
                        "reason": "Reports consume Beta result objects.",
                        "confidence": 0.81,
                        "evidence": ["Gamma consumes model results produced by Beta."],
                    }
                ],
                "uncertain_edges": [
                    {
                        "source_hint": "ReqA::Alpha",
                        "target_hint": "ReqB::Gamma",
                        "reason": "Possible reporting-time metadata dependency, but not clearly must-have.",
                    }
                ],
            }

    agent = DependencyGraphAgent(api_config={"api_key": "test"})
    agent.llm_client = _FakeLLM()

    result = agent.build_requirement_dependency_edges(
        components,
        constraints={
            "must_use_only_component_ids": True,
            "disallow_self_dependency": True,
            "allow_same_requirement_edges": False,
        },
        requirement_edges=[],
    )

    assert len(result["component_edges"]) == 1
    assert result["component_edges"][0]["source"] == "ReqB::Beta"
    assert result["component_edges"][0]["target"] == "ReqA::Alpha"
    assert result["component_edges"][0]["dependency_type"] == "api_call"
    assert len(result["same_requirement_component_edges"]) == 1
    assert result["same_requirement_component_edges"][0]["source"] == "ReqB::Gamma"
    assert len(result["uncertain_edges"]) == 1
    assert len(result["edges"]) == 1
    assert result["edges"][0]["source"] == "ReqB"
    assert result["edges"][0]["target"] == "ReqA"
