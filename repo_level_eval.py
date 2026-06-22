#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path


def load_eval_module():
    module_path = Path(__file__).resolve().parent / "scripts" / "repo_level_eval.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing eval module: {module_path}")
    spec = importlib.util.spec_from_file_location("repo_level_eval_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Unable to load eval module.")
    spec.loader.exec_module(module)
    return module


def read_json_or_jsonl(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def requirements_to_texts(data):
    if isinstance(data, dict) and "requirements" in data:
        data = data["requirements"]
    if not isinstance(data, list):
        raise ValueError("Requirements data must be a list.")
    texts = []
    for item in data:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            desc = str(item.get("description", "")).strip()
            if name and desc:
                text = f"{name}: {desc}"
            else:
                text = name or desc
        else:
            text = ""
        if text:
            texts.append(text)
    return texts


def find_text_items_file(repo_dir, kind):
    repo_dir = Path(repo_dir)
    if kind == "categories":
        preferred = [
            "readme_output/requirements.json",
            "readme_output/requirements.jsonl",
            "agents_output/requirements.json",
            "agents_output/requirements.jsonl",
            "requirements.json",
            "requirements.jsonl",
            "categories.json",
            "categories.jsonl",
        ]
    else:
        preferred = [
            "agents_output/requirements.json",
            "agents_output/requirements.jsonl",
            "readme_output/requirements.json",
            "readme_output/requirements.jsonl",
            "features.json",
            "features.jsonl",
            "requirements.json",
            "requirements.jsonl",
        ]

    for rel in preferred:
        candidate = repo_dir / rel
        if candidate.exists():
            return candidate

    external = find_external_text_items(repo_dir.name, kind)
    if external is not None:
        return external

    fallback_names = [
        "requirements.json",
        "requirements.jsonl",
        "categories.json",
        "categories.jsonl",
        "features.json",
        "features.jsonl",
    ]
    matches = []
    for name in fallback_names:
        matches.extend(repo_dir.rglob(name))
    if matches:
        return sorted(matches)[0]
    return None


def find_external_text_items(repo_name, kind):
    script_dir = Path(__file__).resolve().parent
    roots = [
        script_dir / "repos_v3",
        script_dir / "repos_v2",
        script_dir / "repos_v2 copy",
        script_dir / "repos_v1",
        script_dir / "repos",
    ]
    if kind == "categories":
        preferred = [
            "readme_output/requirements.json",
            "readme_output/requirements.jsonl",
            "agents_output/requirements.json",
            "agents_output/requirements.jsonl",
        ]
    else:
        preferred = [
            "agents_output/requirements.json",
            "agents_output/requirements.jsonl",
            "readme_output/requirements.json",
            "readme_output/requirements.jsonl",
        ]
    for root in roots:
        base = root / repo_name
        if not base.exists():
            continue
        for rel in preferred:
            candidate = base / rel
            if candidate.exists():
                return candidate
    return None


def load_text_items_from_repo(repo_dir, provided_path, kind, eval_utils, text_field=None):
    if provided_path:
        path = Path(provided_path)
    else:
        path = find_text_items_file(repo_dir, kind)
    if path is None:
        raise FileNotFoundError(
            f"Unable to locate {kind} file in {repo_dir}. Provide --{kind}."
        )
    if path.name.startswith("requirements.") and path.suffix in (".json", ".jsonl"):
        data = read_json_or_jsonl(path)
        texts = requirements_to_texts(data)
    else:
        texts = eval_utils.load_text_items(path, text_field=text_field or None)
    return texts, path


def main():
    parser = argparse.ArgumentParser(
        description="Repo-level evaluation from ground-truth and generated repos."
    )
    parser.add_argument("--ground_truth_repo", required=True, help="Ground-truth repo.")
    parser.add_argument("--target_repo", required=True, help="Generated repo.")
    parser.add_argument("--categories", default="", help="Override categories file.")
    parser.add_argument("--features", default="", help="Override features file.")
    parser.add_argument("--tasks_file", default="", help="Task results JSON/JSONL.")
    parser.add_argument("--override_map", default="", help="Optional JSON mapping.")
    parser.add_argument("--text_field", default="", help="Field name for text in JSON.")
    parser.add_argument("--ood_threshold", type=float, default=0.25)
    parser.add_argument(
        "--unsupervised",
        action="store_true",
        help="Use unsupervised k-means on features instead of fixed centroids.",
    )
    parser.add_argument(
        "--exclude_dirs",
        default="tests,test,examples,example,benchmarks,benchmark,docs,doc,build,dist,.git,__pycache__,.venv,venv",
    )
    parser.add_argument("--voting_key", default="")
    parser.add_argument("--success_key", default="")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    args = parser.parse_args()

    eval_utils = load_eval_module()

    ground_truth_repo = Path(args.ground_truth_repo)
    target_repo = Path(args.target_repo)
    if not ground_truth_repo.exists():
        raise FileNotFoundError(f"Missing ground-truth repo: {ground_truth_repo}")
    if not target_repo.exists():
        raise FileNotFoundError(f"Missing target repo: {target_repo}")

    if not args.unsupervised and not args.features:
        raise ValueError("Supervised mode requires --features to be provided.")

    categories, categories_path = load_text_items_from_repo(
        ground_truth_repo,
        args.categories or None,
        "categories",
        eval_utils,
        text_field=args.text_field or None,
    )
    features, features_path = load_text_items_from_repo(
        target_repo,
        args.features or None,
        "features",
        eval_utils,
        text_field=args.text_field or None,
    )

    assignments = eval_utils.assign_categories(
        categories,
        features,
        ood_threshold=args.ood_threshold,
        unsupervised=args.unsupervised,
    )
    overrides = None
    if args.override_map:
        with open(args.override_map, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        assignments = eval_utils.apply_overrides(assignments, overrides)

    coverage = eval_utils.compute_coverage(categories, assignments)
    novelty = eval_utils.compute_novelty(assignments)

    tasks = eval_utils.load_tasks(args.tasks_file) if args.tasks_file else []
    accuracy = eval_utils.compute_accuracy(
        tasks,
        voting_key=args.voting_key or None,
        success_key=args.success_key or None,
    )

    exclude_dirs = {d.strip().lower() for d in args.exclude_dirs.split(",") if d.strip()}
    code_stats = eval_utils.normalized_code_stats(target_repo, exclude_dirs)

    result = {
        "coverage": coverage,
        "novelty": novelty,
        "accuracy": accuracy,
        "code_stats": code_stats,
        "assignments": [
            {"feature": feature, "category": category, "score": score}
            for feature, category, score in assignments
        ],
        "sources": {
            "categories": str(categories_path),
            "features": str(features_path),
        },
    }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=True, indent=2)
    else:
        print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
