#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
TEST_REPO_ROOT = Path(os.environ.get("REPO0_TEST_REPO_ROOT", ROOT / "tmp" / "test_repo")).resolve()
GITHUB_LINK_RE = re.compile(
    r"https://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)/(pull|issues)/(?P<number>\d+)"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


TOKEN_MOD = _load_module(ROOT / "scripts" / "count_python_tokens.py", "count_python_tokens_mod")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agents.infra.llm_client import LLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate requirement iteration outputs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="Repo directory or agents_output directory for the iteration run.")
    parser.add_argument("--final-run-dir", type=Path, default=None, help="Optional cumulative/final run directory for pass_all.")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--synthesize-missing-tests", action="store_true")
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--reasoning-effort", type=str, default="", help="Optional reasoning_effort to pass through to the LLM API.")
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def tokenize_text(text: str) -> List[str]:
    return [tok for tok in normalize_text(text).split() if len(tok) >= 3]


def token_overlap(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa = set(a)
    sb = set(b)
    inter = sa & sb
    if not inter:
        return 0.0
    return len(inter) / max(1, min(len(sa), len(sb)))


def resolve_agents_output(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    if (run_dir / "generated_code").exists():
        return run_dir
    if (run_dir / "agents_output").exists():
        return (run_dir / "agents_output").resolve()
    raise FileNotFoundError(f"Cannot resolve agents_output under {run_dir}")


def load_feature_specs(path: Path) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for row in load_jsonl(path):
        for key, value in row.items():
            if isinstance(value, dict):
                name = value.get("requirement") or value.get("name") or value.get("title") or key
                if not name:
                    continue
                pr_meta = row.get("pr", {}) if isinstance(row.get("pr"), dict) else {}
                specs.append(
                    {
                        "key": key,
                        "name": str(name).strip(),
                        "summary": str(value.get("summary", "") or ""),
                        "details": str(value.get("details", "") or ""),
                        "affected_scope": str(value.get("affected_scope", "") or ""),
                        "requirement_type": str(value.get("requirement_type", "") or ""),
                        "pr": pr_meta,
                    }
                )
    return specs


def load_dag_names(agents_output: Path) -> set[str]:
    dag_path = agents_output / "requirement_dag.json"
    if not dag_path.exists():
        return set()
    data = json.loads(dag_path.read_text(encoding="utf-8"))
    nodes = data.get("nodes", {}) if isinstance(data, dict) else {}
    return {str(name).strip() for name in nodes.keys()}


def compute_pass_rate(feature_names: List[str], dag_names: set[str]) -> Dict[str, Any]:
    if not feature_names:
        return {"total": 0, "passed": 0, "rate": 0.0, "missing": []}
    passed = [name for name in feature_names if name in dag_names]
    missing = [name for name in feature_names if name not in dag_names]
    return {
        "total": len(feature_names),
        "passed": len(passed),
        "rate": len(passed) / len(feature_names),
        "missing": missing,
    }


def load_generated_entries(agents_output: Path) -> List[Dict[str, Any]]:
    path = agents_output / "generated_files.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def load_component_realization(agents_output: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    path = agents_output / "component_realization_report.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    components = data.get("components", []) if isinstance(data, dict) else []
    result: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in components:
        if not isinstance(item, dict):
            continue
        key = (str(item.get("task", "")), str(item.get("component", "")))
        result[key] = item
    return result


def load_component_import_postcheck(agents_output: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    path = agents_output / "component_import_postcheck_report.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("components", []) if isinstance(data, dict) else data if isinstance(data, list) else []
    result: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        key = (str(item.get("task", "")), str(item.get("component", "")))
        result[key] = item
    return result


def extract_changed_tests(pr_data_path: Path) -> List[Dict[str, Any]]:
    test_entries: List[Dict[str, Any]] = []
    for item in load_jsonl(pr_data_path):
        diff = str(item.get("diff", "") or "")
        changed = sorted(set(re.findall(r"^diff --git a/(.+) b/(.+)$", diff, flags=re.MULTILINE)))
        test_files = []
        for left, right in changed:
            target = right or left
            if "/test" in target or target.startswith("tests/") or target.endswith("_test.py") or target.startswith("test_"):
                test_files.append(target)
        linked_refs = []
        desc = str(item.get("description", "") or "")
        for match in GITHUB_LINK_RE.finditer(desc):
            linked_refs.append(
                {
                    "repository": f"{match.group('owner')}/{match.group('repo')}",
                    "number": int(match.group("number")),
                    "url": match.group(0),
                }
            )
        test_entries.append(
            {
                "repository": item.get("repository"),
                "pr_number": item.get("pr_number"),
                "title": item.get("title", ""),
                "changed_test_files": test_files,
                "has_pr_tests": bool(test_files),
                "linked_github_refs": linked_refs,
                "needs_llm_test_synthesis": not bool(test_files),
            }
        )
    return test_entries


def extract_diff_features(diff_text: str) -> Dict[str, Any]:
    changed_files: List[str] = []
    added_lines: List[str] = []
    for line in (diff_text or "").splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                changed_files.append(parts[3][2:] if parts[3].startswith("b/") else parts[3])
            continue
        if line.startswith("+++ ") or line.startswith("--- "):
            continue
        if line.startswith("+"):
            added_lines.append(line[1:])
    identifier_counts: Dict[str, int] = {}
    for line in added_lines:
        for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", line):
            ident_l = ident.lower()
            if ident_l in {"self", "true", "false", "none", "return", "class", "def", "from", "import"}:
                continue
            identifier_counts[ident_l] = identifier_counts.get(ident_l, 0) + 1
    top_identifiers = [name for name, _count in sorted(identifier_counts.items(), key=lambda kv: kv[1], reverse=True)[:40]]
    return {
        "changed_files": changed_files,
        "added_lines": added_lines,
        "top_identifiers": top_identifiers,
    }


def count_python_tokens(generated_code_dir: Path) -> Dict[str, Any]:
    encoder, tokenizer_name = TOKEN_MOD.build_encoder("gpt-4o-mini", "")
    rows, summary = TOKEN_MOD.count_tokens_for_files(generated_code_dir, encoder)
    return {
        "tokenizer": tokenizer_name,
        "summary": summary,
        "top_files": sorted(rows, key=lambda x: int(x["tokens"]), reverse=True)[:20],
    }


def build_generated_code_index(generated_code_dir: Path) -> List[Dict[str, Any]]:
    index: List[Dict[str, Any]] = []
    for py_file in sorted(generated_code_dir.rglob("*.py")):
        try:
            text = py_file.read_text(encoding="utf-8")
        except Exception:
            continue
        identifiers: List[str] = []
        public_symbols: List[str] = []
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    identifiers.append(node.name.lower())
                    if not node.name.startswith("_"):
                        public_symbols.append(node.name)
                elif isinstance(node, ast.Name):
                    identifiers.append(node.id.lower())
                elif isinstance(node, ast.Attribute):
                    identifiers.append(node.attr.lower())
        except Exception:
            identifiers.extend([m.group(0).lower() for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)])
        file_tokens = tokenize_text(py_file.stem.replace("_", " ") + " " + text[:8000])
        index.append(
            {
                "path": str(py_file),
                "relpath": str(py_file.relative_to(generated_code_dir)),
                "tokens": file_tokens,
                "identifiers": sorted(set(identifiers)),
                "public_symbols": sorted(set(public_symbols)),
            }
        )
    return index


def score_generated_entry_for_feature(feature: Dict[str, Any], entry: Dict[str, Any]) -> float:
    feature_tokens = tokenize_text(
        " ".join(
            [
                feature.get("name", ""),
                feature.get("summary", ""),
                feature.get("details", ""),
                feature.get("affected_scope", ""),
            ]
        )
    )
    entry_text = " ".join(
        [
            str(entry.get("parent_task", "") or ""),
            str(entry.get("component", "") or ""),
            str(entry.get("planned_file_path", "") or ""),
            " ".join(str(x) for x in (entry.get("sub_tasks", []) or [])),
            " ".join(str(x) for x in (entry.get("component_responsibilities", []) or [])),
        ]
    )
    entry_tokens = tokenize_text(entry_text)
    return token_overlap(feature_tokens, entry_tokens)


def score_code_file_for_feature(
    feature: Dict[str, Any],
    *,
    diff_features: Dict[str, Any],
    code_info: Dict[str, Any],
) -> float:
    feature_tokens = tokenize_text(
        " ".join([feature.get("name", ""), feature.get("summary", ""), feature.get("details", ""), feature.get("affected_scope", "")])
    )
    path_tokens = tokenize_text(code_info.get("relpath", "").replace("/", " ").replace("_", " "))
    token_score = token_overlap(feature_tokens, code_info.get("tokens", []))
    path_score = token_overlap(feature_tokens, path_tokens)
    diff_ident_score = token_overlap(diff_features.get("top_identifiers", []), code_info.get("identifiers", []))
    return max(token_score, path_score * 0.8, diff_ident_score)


def analyze_feature_implementation(
    features: List[Dict[str, Any]],
    *,
    pr_rows: List[Dict[str, Any]],
    agents_output: Path,
    final_agents_output: Path | None,
) -> Dict[str, Any]:
    dag_names = load_dag_names(agents_output)
    final_dag_names = load_dag_names(final_agents_output) if final_agents_output else set()
    generated_entries = load_generated_entries(agents_output)
    final_generated_entries = load_generated_entries(final_agents_output) if final_agents_output else []
    realization = load_component_realization(agents_output)
    final_realization = load_component_realization(final_agents_output) if final_agents_output else {}
    import_checks = load_component_import_postcheck(agents_output)
    final_import_checks = load_component_import_postcheck(final_agents_output) if final_agents_output else {}
    code_index = build_generated_code_index(agents_output / "generated_code")
    final_code_index = build_generated_code_index(final_agents_output / "generated_code") if final_agents_output else []
    pr_map = {(row.get("repository"), row.get("pr_number")): row for row in pr_rows}

    def _evaluate_one(
        feature: Dict[str, Any],
        *,
        dag: set[str],
        entries: List[Dict[str, Any]],
        realization_map: Dict[Tuple[str, str], Dict[str, Any]],
        import_map: Dict[Tuple[str, str], Dict[str, Any]],
        code_idx: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        pr_meta = feature.get("pr", {}) if isinstance(feature.get("pr"), dict) else {}
        pr_source = pr_map.get((pr_meta.get("repository"), pr_meta.get("pr_number")), {})
        diff_features = extract_diff_features(str(pr_source.get("diff", "") or ""))
        dag_hit = feature["name"] in dag

        scored_entries = []
        for entry in entries:
            score = score_generated_entry_for_feature(feature, entry)
            if score >= 0.22:
                key = (str(entry.get("parent_task", "")), str(entry.get("component", "")))
                scored_entries.append(
                    {
                        "score": round(score, 4),
                        "component": entry.get("component", ""),
                        "parent_task": entry.get("parent_task", ""),
                        "planned_file_path": entry.get("planned_file_path", ""),
                        "realization_passed": bool(realization_map.get(key, {}).get("passed", False)),
                        "import_postcheck_passed": bool(import_map.get(key, {}).get("passed", False)),
                    }
                )
        scored_entries.sort(key=lambda item: item["score"], reverse=True)

        scored_files = []
        for file_info in code_idx:
            score = score_code_file_for_feature(feature, diff_features=diff_features, code_info=file_info)
            if score >= 0.18:
                scored_files.append(
                    {
                        "score": round(score, 4),
                        "relpath": file_info["relpath"],
                        "public_symbols": file_info["public_symbols"][:8],
                    }
                )
        scored_files.sort(key=lambda item: item["score"], reverse=True)

        strong_entry = any(
            item["score"] >= 0.30 and (item["realization_passed"] or item["import_postcheck_passed"])
            for item in scored_entries
        )
        strong_file = any(item["score"] >= 0.26 for item in scored_files)
        implemented = bool(dag_hit and (strong_entry or strong_file))
        return {
            "feature": feature["name"],
            "pr_repository": pr_meta.get("repository"),
            "pr_number": pr_meta.get("pr_number"),
            "dag_hit": dag_hit,
            "implemented": implemented,
            "matched_components": scored_entries[:5],
            "matched_code_files": scored_files[:5],
            "diff_changed_files": diff_features["changed_files"][:10],
            "diff_top_identifiers": diff_features["top_identifiers"][:20],
        }

    pass_one_items = [
        _evaluate_one(
            feature,
            dag=dag_names,
            entries=generated_entries,
            realization_map=realization,
            import_map=import_checks,
            code_idx=code_index,
        )
        for feature in features
    ]

    pass_all_items = None
    if final_agents_output:
        pass_all_items = [
            _evaluate_one(
                feature,
                dag=final_dag_names,
                entries=final_generated_entries,
                realization_map=final_realization,
                import_map=final_import_checks,
                code_idx=final_code_index,
            )
            for feature in features
        ]

    def _summarize(items: List[Dict[str, Any]] | None) -> Dict[str, Any] | None:
        if items is None:
            return None
        total = len(items)
        passed = sum(1 for item in items if item.get("implemented"))
        return {
            "total": total,
            "passed": passed,
            "rate": (passed / total) if total else 0.0,
            "missing": [item["feature"] for item in items if not item.get("implemented")],
            "items": items,
        }

    return {
        "pass_one": _summarize(pass_one_items),
        "pass_all": _summarize(pass_all_items),
    }


def synthesize_missing_fail2pass_tests(
    items: List[Dict[str, Any]],
    *,
    run_agents_output: Path,
    model: str,
    base_url: str,
    api_key: str,
) -> Dict[str, Any]:
    synth_dir = run_agents_output / "iteration_eval" / "fail2pass_synthesized_tests"
    synth_dir.mkdir(parents=True, exist_ok=True)
    llm_client = LLMClient(
        {"api_key": api_key, "base_url": base_url, "model": model, "reasoning_effort": reasoning_effort},
        str(synth_dir),
        agent_name="fail2pass_test_synthesizer",
    )
    results: List[Dict[str, Any]] = []
    for item in items:
        if not item.get("needs_llm_test_synthesis"):
            continue
        pr_number = item.get("pr_number")
        repo = item.get("repository", "")
        description = item.get("description_with_link_context") or item.get("description") or ""
        prompt = f"""
You are generating a fail-to-pass regression test for a Python repository after a feature PR.

Repository: {repo}
PR number: {pr_number}
Title: {item.get("title", "")}
Description:
{description}

Diff:
{item.get("diff", "")}

Return a strict JSON object:
{{
  "test_file": "tests/test_<short_name>.py",
  "test_purpose": "...",
  "test_code": "full pytest file content"
}}

Constraints:
- Generate only one focused regression test file.
- Use pytest.
- The test should fail before the PR change and pass after it.
- Do not use placeholders.
- Keep imports and assertions concrete.
"""
        try:
            payload = llm_client.call_json(
                messages=[
                    {
                        "role": "system",
                        "content": "You generate precise Python regression tests and return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=32768,
                operation_name="fail2pass_test_synthesis",
            )
        except Exception as exc:
            results.append(
                {
                    "repository": repo,
                    "pr_number": pr_number,
                    "synthesized": False,
                    "error": str(exc),
                }
            )
            continue

        test_file = str(payload.get("test_file", "")).strip() or f"tests/test_pr_{pr_number}.py"
        test_code = str(payload.get("test_code", "")).rstrip()
        target_path = synth_dir / f"pr_{pr_number}_{Path(test_file).name}"
        target_path.write_text(test_code + "\n", encoding="utf-8")
        results.append(
            {
                "repository": repo,
                "pr_number": pr_number,
                "synthesized": True,
                "test_file": test_file,
                "artifact_path": str(target_path),
                "test_purpose": payload.get("test_purpose", ""),
            }
        )
    return {
        "artifact_dir": str(synth_dir),
        "items": results,
        "synthesized_count": sum(1 for item in results if item.get("synthesized")),
        "failed_count": sum(1 for item in results if not item.get("synthesized")),
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    run_agents_output = resolve_agents_output(args.run_dir)
    final_agents_output = resolve_agents_output(args.final_run_dir) if args.final_run_dir else None

    evolve_requirements_path = Path(manifest["evolve_requirements_file"]).resolve()
    pr_data_path = Path(manifest["selected_pr_data"]).resolve()
    feature_specs = load_feature_specs(evolve_requirements_path)
    feature_names = [item["name"] for item in feature_specs]
    feature_count = len(feature_specs)

    pr_rows = load_jsonl(pr_data_path)
    implementation_analysis = analyze_feature_implementation(
        feature_specs,
        pr_rows=pr_rows,
        agents_output=run_agents_output,
        final_agents_output=final_agents_output,
    )
    token_stats = count_python_tokens(run_agents_output / "generated_code")
    raw_fail2pass = extract_changed_tests(pr_data_path)
    enriched_fail2pass: List[Dict[str, Any]] = []
    pr_rows_map = {(item.get("repository"), item.get("pr_number")): item for item in pr_rows}
    for item in raw_fail2pass:
        source = pr_rows_map.get((item.get("repository"), item.get("pr_number")), {})
        merged = dict(item)
        merged["description"] = source.get("description", "")
        merged["description_with_link_context"] = source.get("description_with_link_context", "")
        merged["diff"] = source.get("diff", "")
        enriched_fail2pass.append(merged)

    synthesis = None
    if args.synthesize_missing_tests:
        if not args.base_url or not args.api_key:
            raise SystemExit("--synthesize-missing-tests requires --base-url and --api-key")
        synthesis = synthesize_missing_fail2pass_tests(
            enriched_fail2pass,
            run_agents_output=run_agents_output,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
        )

    payload = {
        "manifest": str(args.manifest.resolve()),
        "run_dir": str(run_agents_output),
        "final_run_dir": str(final_agents_output) if final_agents_output else None,
        "feature_complexity": {
            "feature_count": feature_count,
            "python_token_summary": token_stats["summary"],
            "tokenizer": token_stats["tokenizer"],
            "top_files": token_stats["top_files"],
        },
        "implementation_rate": {
            "pass_one": implementation_analysis["pass_one"],
            "pass_all": implementation_analysis["pass_all"],
        },
        "fail2pass": {
            "items": enriched_fail2pass,
            "prs_with_changed_tests": sum(1 for item in enriched_fail2pass if item["has_pr_tests"]),
            "prs_requiring_llm_test_synthesis": sum(1 for item in enriched_fail2pass if item["needs_llm_test_synthesis"]),
            "llm_synthesis": synthesis,
        },
    }

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
