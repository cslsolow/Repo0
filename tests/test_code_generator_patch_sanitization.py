import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.coding.code_generator import CodeGeneratorAgent


def test_strip_patch_artifacts_removes_patch_markers_only():
    source = (
        "*** Begin Patch\n"
        "*** Update File: tests/test_demo.py\n"
        "@@\n"
        "def test_demo():\n"
        "    assert True\n"
        "*** End Patch\n"
    )
    cleaned = CodeGeneratorAgent._strip_patch_artifacts(source)
    assert "*** End Patch" not in cleaned
    assert "*** Update File:" not in cleaned
    assert "@@" not in cleaned
    assert "def test_demo():" in cleaned


def test_find_patch_artifact_issues_reports_marker_lines():
    agent = CodeGeneratorAgent(api_config={})
    issues = agent._find_patch_artifact_issues(
        "def ok():\n    return 1\n*** End Patch\n",
        "tests/test_demo.py",
    )
    assert any("patch artifact marker remains" in item for item in issues)


def test_generate_code_retries_skeleton_path_with_feedback_before_fallback():
    class RetryAgent(CodeGeneratorAgent):
        def __init__(self) -> None:
            super().__init__(api_config={"api_key": "dummy", "codegen_max_retry_times": 5, "skeleton_review_max_retries": 1})
            self.skeleton_feedbacks = []
            self.single_shot_called = False

        def _generate_code_skeleton_tdd(self, *args, **kwargs):
            self.skeleton_feedbacks.append(kwargs.get("previous_attempt_feedback", ""))
            if len(self.skeleton_feedbacks) == 1:
                raise RuntimeError("Skeleton review found unresolved signature structure issues")
            return {"file_path": "pkg/demo.py", "code": "def ok():\n    return 1\n"}

        def _generate_code_single_shot(self, *args, **kwargs):
            self.single_shot_called = True
            return {"file_path": "pkg/fallback.py", "code": "def fallback():\n    return 1\n"}

    agent = RetryAgent()
    result = agent.generate_code(
        component={"name": "DemoComponent", "responsibilities": ["Provide demo behavior"]},
        requirement={"name": "DemoRequirement", "description": "demo"},
        architecture={"components": []},
        language="python",
    )

    assert result["file_path"] == "pkg/demo.py"
    assert agent.single_shot_called is False
    assert len(agent.skeleton_feedbacks) == 2
    assert agent.skeleton_feedbacks[0] == ""
    assert "Skeleton/TDD attempt 1 failed for component 'DemoComponent'" in agent.skeleton_feedbacks[1]


def test_generate_code_uses_dedicated_skeleton_retry_budget():
    class RetryBudgetAgent(CodeGeneratorAgent):
        def __init__(self) -> None:
            super().__init__(
                api_config={
                    "api_key": "dummy",
                    "codegen_max_retry_times": 5,
                    "skeleton_review_max_retries": 2,
                }
            )
            self.skeleton_calls = 0
            self.single_shot_called = False

        def _generate_code_skeleton_tdd(self, *args, **kwargs):
            self.skeleton_calls += 1
            raise RuntimeError(f"skeleton failure {self.skeleton_calls}")

        def _generate_code_single_shot(self, *args, **kwargs):
            self.single_shot_called = True
            return {"file_path": "pkg/fallback.py", "code": "def fallback():\n    return 1\n"}

    agent = RetryBudgetAgent()
    result = agent.generate_code(
        component={"name": "BudgetedComponent", "responsibilities": ["Provide demo behavior"]},
        requirement={"name": "DemoRequirement", "description": "demo"},
        architecture={"components": []},
        language="python",
    )

    assert result["file_path"] == "pkg/fallback.py"
    assert agent.single_shot_called is True
    assert agent.skeleton_calls == 3


def test_normalize_file_path_does_not_apply_repo_specific_remap():
    agent = CodeGeneratorAgent(api_config={"repo": "statsmodels"})
    normalized = agent.normalize_file_path("statsmodels/time_series/demo.py", language="python")
    assert normalized == "statsmodels/time_series/demo.py"


def test_forbidden_peer_repo_roots_default_excludes_current_repo_and_statsmodels():
    agent = CodeGeneratorAgent(api_config={"repo": "statsmodels"})
    roots = agent._forbidden_peer_repo_roots()
    assert "statsmodels" not in roots
    assert roots == {"django", "sklearn", "scikit_learn"}


def test_peer_repo_constraint_text_uses_configured_roots():
    agent = CodeGeneratorAgent(api_config={"repo": "statsmodels", "peer_framework_roots": ["torchvision", "django"]})
    text = agent._peer_repo_constraint_text(include_generic_utils_note=False)
    assert "`torchvision`" in text
    assert "`django`" in text
    assert "statsmodels" not in text


def test_select_validation_patch_updated_files_prefers_diff_only_when_it_clears_issues():
    class StubAgent(CodeGeneratorAgent):
        def _autofix_python_syntax(self, code, component_name, rel_path):
            return code
        def _merge_postprocess_python_issues(self, **kwargs):
            code = kwargs.get("impl_code", "")
            return ["still bad"] if "DEMO VALIDATION FAILURE" in code else []

    agent = StubAgent(api_config={})
    patch = {
        "diff_updated_files": {"pkg/demo.py": "def ok():\n    return 1\n"},
        "full_file_updated_files": {"pkg/demo.py": "def ok():\n    return 2\n"},
        "updated_files": {"pkg/demo.py": "def ok():\n    return 2\n"},
    }

    selected = agent._select_validation_patch_updated_files(
        patch=patch,
        component_name="Demo",
        responsibilities=[],
        rel_impl="pkg/demo.py",
        impl_code='def ok():\n    raise NotImplementedError("DEMO VALIDATION FAILURE")\n',
        rel_test="",
        test_code="",
        implemented_components_context="",
        stage="unit_test",
    )

    assert selected["pkg/demo.py"] == "def ok():\n    return 1\n"


def test_select_validation_patch_updated_files_falls_back_to_full_file_when_diff_leaves_issues():
    class StubAgent(CodeGeneratorAgent):
        def _autofix_python_syntax(self, code, component_name, rel_path):
            return code
        def _merge_postprocess_python_issues(self, **kwargs):
            code = kwargs.get("impl_code", "")
            return ["still bad"] if "DEMO VALIDATION FAILURE" in code else []

    agent = StubAgent(api_config={})
    patch = {
        "diff_updated_files": {"pkg/demo.py": 'def ok():\n    raise NotImplementedError("DEMO VALIDATION FAILURE")\n'},
        "full_file_updated_files": {"pkg/demo.py": "def ok():\n    return 2\n"},
        "updated_files": {"pkg/demo.py": "def ok():\n    return 2\n"},
    }

    selected = agent._select_validation_patch_updated_files(
        patch=patch,
        component_name="Demo",
        responsibilities=[],
        rel_impl="pkg/demo.py",
        impl_code='def ok():\n    raise NotImplementedError("DEMO VALIDATION FAILURE")\n',
        rel_test="",
        test_code="",
        implemented_components_context="",
        stage="unit_test",
    )

    assert selected["pkg/demo.py"] == "def ok():\n    return 2\n"


def test_derive_generation_status_marks_retained_after_tdd_failure():
    agent = CodeGeneratorAgent(api_config={})
    status = agent._derive_generation_status(
        {
            "component_name": "Demo",
            "file_path": "pkg/demo.py",
            "code": "def demo():\n    return 1\n",
            "skeleton_fill_tdd": {"final_pytest_rc": 1},
        }
    )
    assert status == "retained_after_tdd_failure"


def test_extract_component_metadata_uses_retained_tdd_failure_status():
    agent = CodeGeneratorAgent(api_config={})
    metadata = agent.extract_component_metadata(
        {
            "component_name": "Demo",
            "file_path": "pkg/demo.py",
            "code": "def demo():\n    return 1\n",
            "skeleton_fill_tdd": {"final_pytest_rc": 1},
            "generation_status": "retained_after_tdd_failure",
            "tdd_passed": False,
            "tdd_final_pytest_rc": 1,
        },
        requirement_node="Parent",
    )
    assert metadata["status"] == "retained_after_tdd_failure"
    assert metadata["tdd_passed"] is False
    assert metadata["tdd_final_pytest_rc"] == 1


def test_skeleton_and_test_review_use_dedicated_retry_budgets():
    class StubSkeletonReviewAgent:
        def __init__(self):
            self.calls = 0
        def review_skeleton(self, **kwargs):
            self.calls += 1
            return {"updated_files": {kwargs["planned_file_path"]: kwargs["skeleton_code"]}}
        def extract_reviewed_skeleton_code(self, *, patch_result, planned_file_path, fallback_skeleton_code):
            return fallback_skeleton_code

    class StubTestReviewAgent:
        def __init__(self):
            self.calls = 0
        def review_test(self, **kwargs):
            self.calls += 1
            return {"updated_files": {kwargs["test_file_path"]: kwargs["test_code"]}}
        def extract_reviewed_test_code(self, *, patch_result, test_file_path, fallback_test_code):
            return fallback_test_code

    class ReviewBudgetAgent(CodeGeneratorAgent):
        def __init__(self) -> None:
            super().__init__(api_config={
                "api_key": "dummy",
                "codegen_max_retry_times": 9,
                "skeleton_review_llm_max_retries": 2,
                "test_review_max_retries": 1,
            })
            self._stub_skeleton = StubSkeletonReviewAgent()
            self._stub_test = StubTestReviewAgent()
        def _get_skeleton_review_agent(self):
            return self._stub_skeleton
        def _get_test_review_agent(self):
            return self._stub_test
        def _find_skeleton_responsibility_alignment_issues(self, **kwargs):
            return ["weak"]
        def _find_test_notimplemented_alignment_issues(self, **kwargs):
            return ["bad test"]

    agent = ReviewBudgetAgent()
    skeleton_result = agent._apply_skeleton_review_with_retries(
        component_name="Demo",
        responsibilities=["r1"],
        planned_file_path="pkg/demo.py",
        skeleton_code='def demo():\n    raise NotImplementedError("TDD")\n',
    )
    assert skeleton_result.strip() == 'def demo():\n    raise NotImplementedError("TDD")'.strip()
    assert agent._stub_skeleton.calls == 3

    try:
        agent._apply_test_review_with_retries(
            component_name="Demo",
            responsibilities=["r1"],
            module_qualname="pkg.demo",
            planned_file_path="pkg/demo.py",
            skeleton_code='def demo():\n    raise NotImplementedError("TDD")\n',
            test_file_path="tests/test_demo.py",
            test_code='def test_demo():\n    assert True\n',
        )
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "Test generation/review left placeholder-behavior assertions" in str(exc)
    assert agent._stub_test.calls == 2


def test_skeleton_review_exhaustion_falls_back_to_current_skeleton():
    class StubSkeletonReviewAgent:
        def __init__(self):
            self.calls = 0
        def review_skeleton(self, **kwargs):
            self.calls += 1
            return {"updated_files": {kwargs["planned_file_path"]: kwargs["skeleton_code"]}}
        def extract_reviewed_skeleton_code(self, *, patch_result, planned_file_path, fallback_skeleton_code):
            return fallback_skeleton_code

    class SoftFallbackAgent(CodeGeneratorAgent):
        def __init__(self) -> None:
            super().__init__(api_config={"api_key": "dummy", "skeleton_review_llm_max_retries": 1})
            self._stub_skeleton = StubSkeletonReviewAgent()
        def _get_skeleton_review_agent(self):
            return self._stub_skeleton
        def _find_skeleton_responsibility_alignment_issues(self, **kwargs):
            return ["weak"]

    agent = SoftFallbackAgent()
    skeleton = 'def demo():\n    raise NotImplementedError("TDD")\n'
    result = agent._apply_skeleton_review_with_retries(
        component_name="Demo",
        responsibilities=["r1"],
        planned_file_path="pkg/demo.py",
        skeleton_code=skeleton,
    )
    assert result.strip() == skeleton.strip()
    assert agent._stub_skeleton.calls == 2


def test_save_generated_code_repairs_compile_failure_after_write():
    agent = CodeGeneratorAgent(api_config={})
    with tempfile.TemporaryDirectory() as tmp:
        result = agent.save_generated_code(
            {
                "component_name": "Demo",
                "file_path": "pkg/demo.py",
                "code": (
                    "from typing import Any\n"
                    "class Demo:\n"
                    "    def export/import(self, *args, **kwargs) -> Any:\n"
                    "        return None\n"
                ),
                "tests": {},
                "skeleton_fill_tdd": {"final_pytest_rc": 0},
            },
            tmp,
            create_tests=False,
        )
        assert result["code"].endswith("pkg/demo.py")
        saved = Path(result["code"]).read_text()
        assert "def export_import(self, *args, **kwargs) -> Any:" in saved


def test_save_generated_code_fails_when_compile_postcheck_cannot_repair():
    agent = CodeGeneratorAgent(api_config={})
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "pkg/broken_demo.py"
        result = agent.save_generated_code(
            {
                "component_name": "BrokenDemo",
                "file_path": "pkg/broken_demo.py",
                "code": "def broken(\n",
                "tests": {},
                "skeleton_fill_tdd": {"final_pytest_rc": 0},
            },
            tmp,
            create_tests=False,
        )
        assert result == {"code": str(target)}
        assert target.exists()
        assert agent._evaluate_python_compile_postcheck(target)["passed"] is False


def test_repair_forbidden_peer_repo_usage_rewrites_only_offending_slice():
    class StubPatchAgent:
        def generate_patch(self, **kwargs):
            rel_path, content = next(iter(kwargs["related_files"].items()))
            revised = content.replace(
                "from statsmodels.api import OLS\n\n",
                "def _local_ols(x, y):\n    return (x, y)\n\n",
            ).replace("return OLS(x, y)", "return _local_ols(x, y)")
            return {"updated_files": {rel_path: revised}}

    class PeerRepairAgent(CodeGeneratorAgent):
        def __init__(self) -> None:
            super().__init__(api_config={"api_key": "dummy"})
            self._stub_patch = StubPatchAgent()
        def _get_patch_agent(self):
            return self._stub_patch

    agent = PeerRepairAgent()
    source = (
        "from statsmodels.api import OLS\n\n"
        "def fit(x, y):\n"
        "    return OLS(x, y)\n"
    )
    patched, report = agent._repair_forbidden_peer_repo_usage(
        component_name="PeerDemo",
        rel_path="sklearn/core/demo.py",
        code=source,
        responsibilities=["Fit a simple local model"],
        stage="component_complete",
    )
    assert report["passed"] is True
    assert "from statsmodels.api import OLS" not in patched
    assert "def local_ols" in patched or "def _local_ols" in patched
    assert "return local_ols(x, y)" in patched or "return _local_ols(x, y)" in patched


def test_save_generated_code_runs_peer_repo_postcheck_before_compile():
    class StubPatchAgent:
        def generate_patch(self, **kwargs):
            rel_path, content = next(iter(kwargs["related_files"].items()))
            revised = content.replace(
                "from statsmodels.api import OLS\n\n",
                "def _local_ols(x, y):\n    return (x, y)\n\n",
            ).replace("return OLS(x, y)", "return _local_ols(x, y)")
            return {"updated_files": {rel_path: revised}}

    class PeerRepairAgent(CodeGeneratorAgent):
        def __init__(self) -> None:
            super().__init__(api_config={"api_key": "dummy"})
            self._stub_patch = StubPatchAgent()
        def _get_patch_agent(self):
            return self._stub_patch

    agent = PeerRepairAgent()
    with tempfile.TemporaryDirectory() as tmp:
        result = agent.save_generated_code(
            {
                "component_name": "PeerDemo",
                "file_path": "sklearn/core/demo.py",
                "code": (
                    "from statsmodels.api import OLS\n\n"
                    "def fit(x, y):\n"
                    "    return OLS(x, y)\n"
                ),
                "tests": {},
                "skeleton_fill_tdd": {"final_pytest_rc": 0},
            },
            tmp,
            create_tests=False,
        )
        assert result["code"].endswith("sklearn/core/demo.py")
        saved = Path(result["code"]).read_text()
        assert "from statsmodels.api import OLS" not in saved
        assert "return local_ols(x, y)" in saved or "return _local_ols(x, y)" in saved


def test_normalize_file_path_uses_repo_root_when_repo_is_not_statsmodels():
    agent = CodeGeneratorAgent(api_config={"repo": "pandas", "api_key": "dummy"})
    assert agent.normalize_file_path("core/frame.py") == "pandas/core/frame.py"
    assert agent.normalize_file_path("") == "pandas/generated/unnamed_component.py"
    assert agent._fallback_file_plan([{"name": "Demo Component"}])["Demo Component"] == "pandas/generated/demo_component.py"


def test_postprocess_validation_prompt_uses_dynamic_repo_pattern():
    class CapturePatchAgent:
        def __init__(self) -> None:
            self.task_description = None

        def generate_patch(self, **kwargs):
            self.task_description = kwargs.get("task_description", "")
            return {"updated_files": {k: v for k, v in kwargs.get("related_files", {}).items()}}

    class PromptCaptureAgent(CodeGeneratorAgent):
        def __init__(self) -> None:
            super().__init__(api_config={"repo": "pandas", "api_key": "dummy", "post_generation_max_repair_rounds": 1})
            self._stub_patch = CapturePatchAgent()

        def _get_patch_agent(self):
            return self._stub_patch

        def _raise_if_postprocess_still_fails(self, **kwargs):
            return None

    agent = PromptCaptureAgent()
    agent._postprocess_python_generation(
        component_name="Demo",
        responsibilities=["Provide demo behavior"],
        rel_impl="pandas/core/demo.py",
        impl_code='def demo():\n    raise NotImplementedError("TDD")\n',
        rel_test="tests/test_demo.py",
        test_code="def test_demo():\n    assert True\n",
        implemented_components_context="",
        stage="pre_tdd_reconciliation",
    )

    prompt = agent._stub_patch.task_description
    assert "`pandas.*`" in prompt
    assert "`statsmodels.*`" not in prompt


def test_action_guidance_block_only_emits_rewrite_hint():
    agent = CodeGeneratorAgent(api_config={})

    rewrite_block = agent._action_guidance_block(
        {
            "name": "Demo",
            "recommended_action": "revise",
            "recommended_action_rationale": "Existing API shape is too tangled.",
        }
    )
    save_block = agent._action_guidance_block(
        {
            "name": "Demo",
            "recommended_action": "save",
        }
    )

    assert "strategist marked this component as `revise`" in rewrite_block.lower()
    assert "Revise rationale: Existing API shape is too tangled." in rewrite_block
    assert save_block == ""
