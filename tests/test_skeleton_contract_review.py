import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.coding.code_generator import CodeGeneratorAgent
from agents.coding.skeleton_review_agent import SkeletonReviewAgent
from agents.coding.structured_contracts import extract_structured_contract_facts, find_structured_contract_issues
from agents.cognitive.memory import MemoryAgent, MemorySnapshot


def test_review_python_skeleton_repairs_bare_star_signature():
    agent = CodeGeneratorAgent(api_config={})
    source = (
        "class Rotator:\n"
        "    def rotate(self, method: str, *, **kwargs) -> None:\n"
        "        raise NotImplementedError(\"TDD\")\n"
    )
    reviewed = agent._review_python_skeleton(
        component_name="Rotator",
        rel_path="statsmodels/core/rotator.py",
        code=source,
    )
    compile(reviewed, "<reviewed>", "exec")
    assert "def rotate(self, method: str, **kwargs) -> None:" in reviewed


def test_review_python_skeleton_repairs_comma_after_function_name():
    agent = CodeGeneratorAgent(api_config={})
    source = (
        "from typing import Any\n"
        "class Demo:\n"
        "    def likelihood_ratio,(self, *args, **kwargs) -> Any:\n"
        "        raise NotImplementedError(\"TDD\")\n"
    )
    reviewed = agent._review_python_skeleton(
        component_name="Demo",
        rel_path="statsmodels/core/demo.py",
        code=source,
    )
    compile(reviewed, "<reviewed>", "exec")
    assert "def likelihood_ratio(self, *args, **kwargs) -> Any:" in reviewed


def test_find_signature_structure_issues_flags_illegal_punctuation_after_name():
    agent = CodeGeneratorAgent(api_config={})
    issues = agent._find_signature_structure_issues(
        "class Demo:\n    def serialization/orchestration:(self, *args, **kwargs):\n        return None\n",
        "statsmodels/core/demo.py",
    )
    assert any("illegal characters" in issue or "punctuation immediately after the function name" in issue for issue in issues)


def test_review_python_skeleton_sanitizes_slash_identifier():
    agent = CodeGeneratorAgent(api_config={})
    source = (
        "class Demo:\n"
        "    def serialization/orchestration:(self, *args, **kwargs):\n"
        "        raise NotImplementedError(\"TDD\")\n"
    )
    reviewed = agent._review_python_skeleton(
        component_name="Demo",
        rel_path="statsmodels/core/demo.py",
        code=source,
    )
    compile(reviewed, "<reviewed>", "exec")
    assert "def serialization_orchestration(self, *args, **kwargs):" in reviewed


def test_find_forbidden_peer_repo_imports_flags_core_algorithm_delegation():
    agent = CodeGeneratorAgent(api_config={"repo": "scikit-learn"})
    issues = agent._find_forbidden_peer_repo_imports(
        "from statsmodels.api import OLS\nimport django.db\n",
        "sklearn/core/demo.py",
    )
    assert len(issues) == 2
    assert any("statsmodels.api" in issue for issue in issues)
    assert any("django.db" in issue for issue in issues)


def test_extract_structured_contract_facts_prefers_dict_like_update_without_false_positive():
    source = (
        "class Demo:\n"
        "    def __init__(self):\n"
        "        self._config = {}\n"
        "    def load(self, data):\n"
        "        self._config.update(data)\n"
    )
    facts = extract_structured_contract_facts(source, "pkg/demo.py")
    issues = find_structured_contract_issues(source, "pkg/demo.py")
    assert any("self._config should remain dict-like" in fact for fact in facts)
    assert not any("self._config" in issue for issue in issues)


def test_memory_format_includes_contract_facts():
    memory = MemoryAgent(ROOT)
    memory.snapshot = MemorySnapshot(
        repo_name="statsmodels",
        files=[],
        requirements=[],
        notes="",
    )
    memory.register_component_implementation(
        component_name="DemoConfig",
        requirement_node="ParentRequirement",
        file_path="statsmodels/core/demo_config.py",
        class_names=["DemoConfig"],
        function_signatures=[],
        exports=["DemoConfig"],
        status="implemented",
        structured_contract_facts=["statsmodels/core/demo_config.py: self._config should remain dict-like using methods update"],
    )
    rendered = memory.format_implementations_for_prompt()
    assert "State Contracts:" in rendered
    assert "self._config should remain dict-like" in rendered


def test_find_test_notimplemented_alignment_issues_flags_concrete_api_expectations():
    agent = CodeGeneratorAgent(api_config={})
    skeleton = (
        "class Demo:\n"
        "    def run(self) -> int:\n"
        "        raise NotImplementedError(\"TDD\")\n"
    )
    test_code = (
        "import pytest\n"
        "from pkg.demo import Demo\n\n"
        "def test_run_raises():\n"
        "    with pytest.raises(NotImplementedError):\n"
        "        Demo().run()\n"
    )
    issues = agent._find_test_notimplemented_alignment_issues(
        skeleton_code=skeleton,
        test_code=test_code,
        rel_test="tests/test_demo.py",
    )
    assert any("asserts NotImplementedError for a concrete API" in issue for issue in issues)


def test_find_test_notimplemented_alignment_issues_allows_abstract_interfaces():
    agent = CodeGeneratorAgent(api_config={})
    skeleton = (
        "from abc import ABC, abstractmethod\n"
        "class Demo(ABC):\n"
        "    @abstractmethod\n"
        "    def run(self) -> int:\n"
        "        raise NotImplementedError(\"TDD\")\n"
    )
    test_code = (
        "import pytest\n"
        "from pkg.demo import Demo\n\n"
        "def test_run_raises():\n"
        "    with pytest.raises(NotImplementedError):\n"
        "        Demo().run()\n"
    )
    issues = agent._find_test_notimplemented_alignment_issues(
        skeleton_code=skeleton,
        test_code=test_code,
        rel_test="tests/test_demo.py",
    )
    assert issues == []


def test_find_skeleton_responsibility_alignment_issues_flags_weak_skeleton():
    agent = CodeGeneratorAgent(api_config={})
    issues = agent._find_skeleton_responsibility_alignment_issues(
        component_name="DemoService",
        responsibilities=[
            "Compute calibrated predictions and expose JSON-serializable diagnostics metadata",
        ],
        skeleton_code=(
            "class DemoService:\n"
            "    def run(self, x):\n"
            "        raise NotImplementedError(\"TDD\")\n"
        ),
        rel_path="pkg/demo_service.py",
    )
    assert issues
    assert any("weakly realized" in issue for issue in issues)


def test_skeleton_review_agent_returns_fallback_when_patch_empty():
    agent = SkeletonReviewAgent(api_config={})
    reviewed = agent.extract_reviewed_skeleton_code(
        patch_result={},
        planned_file_path="pkg/demo.py",
        fallback_skeleton_code="class Demo:\n    pass\n",
    )
    assert reviewed == "class Demo:\n    pass\n"
