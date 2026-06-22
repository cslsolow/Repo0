import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.memory import ComponentImplementation, MemoryAgent, MemorySnapshot  # noqa: E402


def test_format_implementations_groups_methods_under_classes():
    agent = MemoryAgent(workspace_root=ROOT)
    component_key = "Req1::ComponentA"
    impl = ComponentImplementation(
        component_name="ComponentA",
        requirement_node="Req1",
        file_path="src/component_a.py",
        class_names=["Foo"],
        function_signatures=[
            {"name": "bar", "params": ["x"], "return_type": "int", "class_name": "Foo"},
            {"name": "__init__", "params": ["self", "x"], "return_type": "None", "class_name": "Foo"},
            {"name": "_private", "params": ["x"], "return_type": "int", "class_name": "Foo"},
            {"name": "util", "params": ["y"], "return_type": "str"},
            {"name": "_helper", "params": ["z"], "return_type": "str"},
            {"name": "__helper", "params": ["z"], "return_type": "str"},
        ],
        status="implemented",
    )
    agent.snapshot = MemorySnapshot(
        repo_name="demo",
        files=[],
        requirements=[],
        notes="",
        implemented_components={component_key: impl},
    )

    output = agent.format_implementations_for_prompt()
    print(output)

    assert "[ComponentA] (Parent: Req1)" in output
    assert "Classes:" in output
    assert "    - Foo" in output
    assert "Methods:" in output
    assert "bar(x) -> int" in output
    assert "__init__(self, x) -> None" in output
    assert "_private(x) -> int" not in output
    assert "Module Functions:" in output
    assert "util(y) -> str" in output
    assert "_helper(z) -> str" not in output
