"""Review and patch generated skeletons before test generation/fill."""

from __future__ import annotations

from typing import Any, Dict, List

from .patch_agent import PatchAgent


class SkeletonReviewAgent:
    """LLM-assisted review pass for generated skeleton modules."""

    def __init__(self, api_config: Dict[str, Any] | None = None, output_dir: str = ".") -> None:
        self.api_config = api_config or {}
        self.output_dir = output_dir
        self.patch_agent = PatchAgent(self.api_config, output_dir)

    def review_skeleton(
        self,
        *,
        component_name: str,
        responsibilities: List[Any],
        planned_file_path: str,
        skeleton_code: str,
        implemented_components_context: str = "",
        previous_attempt_feedback: str = "",
    ) -> Dict[str, Any]:
        responsibilities_block = "\n".join(
            f"- {str(item).strip()}" for item in responsibilities if str(item).strip()
        ) or "- None provided"
        task_description = (
            f"Review and repair the generated skeleton module for component '{component_name}'.\n"
            f"Planned implementation path: {planned_file_path}\n\n"
            "Review goals:\n"
            "1. Ensure the skeleton's public API and docstrings faithfully cover the listed responsibilities.\n"
            "2. Add or adjust public classes/functions/signatures only when needed to realize a responsibility that is currently missing.\n"
            "3. Keep the skeleton minimal, importable, and suitable for later test generation and implementation fill.\n"
            "4. Do not implement production logic yet; preserve explicit TDD placeholders in concrete members.\n"
            "5. Prefer precise docstrings and signatures that make the intended final behavior testable.\n\n"
            "Responsibilities:\n"
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
        }
        return self.patch_agent.generate_patch(
            task_description=task_description,
            related_files=related_files,
            incremental_goal="Review and improve the generated skeleton before tests are written.",
            failure_kind="validation_failure",
            telemetry_context={
                "component_name": component_name,
                "stage": "skeleton_review",
                "planned_file_path": planned_file_path,
                "file_role": "impl_only",
            },
        )

    def extract_reviewed_skeleton_code(
        self,
        *,
        patch_result: Dict[str, Any],
        planned_file_path: str,
        fallback_skeleton_code: str,
    ) -> str:
        updated_files = patch_result.get("updated_files", {}) if isinstance(patch_result, dict) else {}
        if isinstance(updated_files, dict):
            candidate = updated_files.get(planned_file_path)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        return fallback_skeleton_code


__all__ = ["SkeletonReviewAgent"]
