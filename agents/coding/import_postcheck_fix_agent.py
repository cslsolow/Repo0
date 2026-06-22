"""Specialized fix agent for import-postcheck/runtime-import failures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .patch_agent import PatchAgent


class ImportPostcheckFixAgent:
    """Thin wrapper around PatchAgent with a tighter prompt for import smoke failures."""

    def __init__(self, api_config: Dict[str, Any] | None = None, output_dir: str = ".") -> None:
        self.api_config = api_config or {}
        self.output_dir = output_dir
        self.patch_agent = PatchAgent(self.api_config, output_dir)

    def collect_related_files(
        self,
        repo_root: str | Path,
        file_paths: List[str],
        max_file_chars: int = 20000,
    ) -> Dict[str, str]:
        return self.patch_agent.collect_related_files(
            repo_root=repo_root,
            file_paths=file_paths,
            max_file_chars=max_file_chars,
        )

    def write_files(self, repo_root: str | Path, updated_files: Dict[str, str]) -> List[str]:
        return self.patch_agent.write_files(repo_root, updated_files)

    def fix_import_failure(
        self,
        *,
        component_name: str,
        module_name: str,
        planned_file_path: str,
        import_error: str,
        related_files: Dict[str, str],
        implemented_components_context: str = "",
    ) -> Dict[str, Any]:
        task_description = (
            f"Repair a Python import smoke failure for component '{component_name}'.\n"
            f"Target module: {module_name}\n"
            f"Planned implementation path: {planned_file_path}\n\n"
            "Goal:\n"
            "Make `importlib.import_module(target_module)` succeed.\n\n"
            "Constraints:\n"
            "1. Fix the import-time/runtime error shown below.\n"
            "2. Keep file paths unchanged.\n"
            "3. Preserve existing public APIs and exported symbol names unless the stack trace proves they are wrong.\n"
            "4. Prefer minimal repairs to top-level initialization, registry/state shape mismatches, and bad imports.\n"
            "5. Do not remove required behavior just to silence the import error.\n"
            "6. Do not introduce placeholder code or NotImplementedError.\n"
            "7. If the error comes from a related file in the stack trace, fix that file directly.\n\n"
            "Import smoke failure:\n"
            f"{import_error[-8000:]}"
        )
        if implemented_components_context:
            task_description += (
                "\n\nIMPLEMENTED COMPONENTS CONTEXT:\n"
                f"{implemented_components_context}"
            )

        return self.patch_agent.generate_patch(
            task_description=task_description,
            related_files=related_files,
            compile_error=import_error[-8000:],
            incremental_goal="Repair import-time/runtime failure reported by module import smoke test.",
            failure_kind="import_failure",
            telemetry_context={
                "component_name": component_name,
                "stage": "import_postcheck",
                "planned_file_path": planned_file_path,
                "module_name": module_name,
                "file_role": "related_files",
            },
        )


__all__ = ["ImportPostcheckFixAgent"]
