"""Review and patch generated tests before implementation fill/TDD."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .patch_agent import PatchAgent


class TestReviewAgent:
    """LLM-assisted review pass for generated pytest files."""

    def __init__(self, api_config: Dict[str, Any] | None = None, output_dir: str = ".") -> None:
        self.api_config = api_config or {}
        self.output_dir = output_dir
        self.patch_agent = PatchAgent(self.api_config, output_dir)

    def review_test(
        self,
        *,
        component_name: str,
        responsibilities: List[Any],
        module_qualname: str,
        planned_file_path: str,
        skeleton_code: str,
        test_file_path: str,
        test_code: str,
        implemented_components_context: str = "",
        previous_attempt_feedback: str = "",
    ) -> Dict[str, Any]:
        responsibilities_block = "\n".join(
            f"- {str(item).strip()}" for item in responsibilities if str(item).strip()
        ) or "- None provided"
        task_description = (
            f"Review and repair the generated pytest file for component '{component_name}'.\n"
            f"Target implementation module: {module_qualname}\n"
            f"Planned implementation path: {planned_file_path}\n"
            f"Test file path: {test_file_path}\n\n"
            "Review goals:\n"
            "1. Ensure imports reference the exact planned module path and symbol names.\n"
            "2. Remove incorrect assumptions about APIs that are not present in the skeleton.\n"
            "3. Keep the tests focused on the component responsibilities and main public surface.\n"
            "4. Do not leave placeholder code, pass-only tests, or malformed pytest constructs.\n"
            "5. Prefer import/module-path corrections, symbol corrections, and expectation tightening.\n"
            "6. Do not rewrite the tests into trivial smoke checks; keep them meaningful and responsibility-driven.\n\n"
            "7. Do not assert that concrete APIs should keep raising NotImplementedError in the final implementation.\n"
            "8. Only keep NotImplementedError expectations for explicitly abstract interfaces (for example ABC/Protocol/abstractmethod).\n\n"
            "Component responsibilities:\n"
            f"{responsibilities_block}\n\n"
            "Skeleton source:\n"
            f"```python\n{skeleton_code}\n```"
        )
        if implemented_components_context:
            task_description += (
                "\n\nIMPLEMENTED COMPONENTS CONTEXT:\n"
                f"{implemented_components_context}"
            )
        if previous_attempt_feedback:
            task_description += (
                "\n\nPREVIOUS ATTEMPT FEEDBACK:\n"
                f"{previous_attempt_feedback}"
            )

        related_files = {
            planned_file_path: skeleton_code,
            test_file_path: test_code,
        }
        return self.patch_agent.generate_patch(
            task_description=task_description,
            related_files=related_files,
            incremental_goal="Review and improve the generated pytest file before implementation fill.",
            failure_kind="validation_failure",
            telemetry_context={
                "component_name": component_name,
                "stage": "test_review",
                "planned_file_path": planned_file_path,
                "module_name": module_qualname,
                "file_role": "impl_and_test",
            },
        )

    def extract_reviewed_test_code(
        self,
        *,
        patch_result: Dict[str, Any],
        test_file_path: str,
        fallback_test_code: str,
    ) -> str:
        updated_files = patch_result.get("updated_files", {}) if isinstance(patch_result, dict) else {}
        if isinstance(updated_files, dict):
            candidate = updated_files.get(test_file_path)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        return fallback_test_code


__all__ = ["TestReviewAgent"]
