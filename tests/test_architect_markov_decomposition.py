import sys
from pathlib import Path

import pytest

pytest.importorskip("openai")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.cognitive.architect import ArchitectAgent  # noqa: E402
from agents.rqmts.dag import RequirementDAG, RequirementNode  # noqa: E402


def test_decompose_requirement_labels_reasoning_into_sub_requirement_dag():
    calls = []

    class FakeLLM:
        def __init__(self):
            self.responses = [
                {
                    "analysis": (
                        "The requirement needs a public API, request validation, "
                        "and standardized result formatting. Formatting depends on "
                        "the API contract."
                    )
                },
                {
                    "analysis": "Segmented API behavior before dependent result formatting.",
                    "sub_requirements": [
                        {
                            "name": "core-api",
                            "description": "Expose the public API and validate request inputs.",
                            "depend": [],
                            "rationale": "The public API defines the entry contract.",
                        },
                        {
                            "name": "result-formatting",
                            "description": "Standardize result objects produced by the public API.",
                            "depend": [0],
                            "rationale": "Result formatting depends on the API contract.",
                        },
                    ],
                },
            ]

        def call_json(self, messages, **_kwargs):
            calls.append(messages[-1]["content"])
            return self.responses[len(calls) - 1]

    architect = ArchitectAgent(api_config={})
    architect.llm_client = FakeLLM()

    result = architect.decompose_requirement(
        RequirementNode(
            name="Unified API",
            description="Provide a unified API with validation and standardized result formatting.",
        )
    )

    assert [item.name for item in result] == [
        "Unified API::core-api",
        "Unified API::result-formatting",
    ]
    assert [item.order for item in result] == [0, 1]
    assert result[0].metadata["depend"] == []
    assert result[1].metadata["depend"] == [0]
    assert "Do not produce sub-requirements yet" in calls[0]
    assert "Complete Requirement Analysis" in calls[1]
    assert "Formatting depends on" in calls[1]


def test_decompose_dag_uses_labeled_dependency_edges():
    class FakeArchitect(ArchitectAgent):
        def __init__(self):
            super().__init__(api_config={})

        def decompose_requirement(self, requirement):
            return [
                self._make_sub(requirement.name, "api", 0, []),
                self._make_sub(requirement.name, "formatting", 1, [0]),
            ]

        def _make_sub(self, parent, name, order, depend):
            from agents.cognitive.architect import SubRequirement

            return SubRequirement(
                name=f"{parent}::{name}",
                description=name,
                parent=parent,
                order=order,
                metadata={"depend": depend},
            )

    dag = RequirementDAG(
        nodes={"Unified API": RequirementNode(name="Unified API", description="d")},
        adjacency={"Unified API": set()},
    )

    decomposed = FakeArchitect().decompose_dag(dag)

    assert decomposed.adjacency["Unified API::api"] == {"Unified API::formatting"}
