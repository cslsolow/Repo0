import argparse
import ast
import json
import math
import re
from io import BytesIO
from pathlib import Path
from tokenize import tokenize, COMMENT, NL, NEWLINE, INDENT, DEDENT, ENCODING, STRING


OOD_LABEL = "__OOD__"


def load_text_items(path, text_field=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix == ".jsonl":
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return normalize_items(items, text_field=text_field)
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return normalize_items(data, text_field=text_field)
    if path.suffix == ".txt":
        with path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def normalize_items(data, text_field=None):
    if isinstance(data, list):
        if not data:
            return []
        if isinstance(data[0], str):
            return data
        if isinstance(data[0], dict):
            return [pick_text(item, text_field) for item in data]
        raise ValueError("Unsupported list contents.")
    if isinstance(data, dict):
        for key in ("categories", "features", "items", "data"):
            if key in data:
                return normalize_items(data[key], text_field=text_field)
        if text_field and text_field in data:
            return normalize_items(data[text_field], text_field=None)
    raise ValueError("Unsupported JSON structure for text items.")


def pick_text(item, text_field=None):
    if text_field and text_field in item:
        return str(item[text_field])
    for key in ("text", "feature", "category", "name", "desc", "description"):
        if key in item:
            return str(item[key])
    raise ValueError(f"Cannot find text field in item: {item}")


def tokenize_text(text):
    return [t for t in re.split(r"[^A-Za-z0-9_]+", text.lower()) if t]


def build_tfidf_vectors(texts):
    docs = [tokenize_text(t) for t in texts]
    df = {}
    for doc in docs:
        for token in set(doc):
            df[token] = df.get(token, 0) + 1
    n_docs = max(len(docs), 1)
    vocab = {token: idx for idx, token in enumerate(sorted(df.keys()))}
    idf = {token: math.log((n_docs + 1) / (df[token] + 1)) + 1 for token in df}
    vectors = []
    for doc in docs:
        tf = {}
        for token in doc:
            tf[token] = tf.get(token, 0) + 1
        vec = [0.0] * len(vocab)
        for token, count in tf.items():
            vec[vocab[token]] = (count / len(doc)) * idf[token]
        vectors.append(vec)
    return vectors


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def mean_vector(vectors, dim):
    if not vectors or dim == 0:
        return [0.0] * dim
    summed = [0.0] * dim
    for vec in vectors:
        for i, val in enumerate(vec):
            summed[i] += val
    return [val / len(vectors) for val in summed]


def kmeans(vectors, k, max_iter=10):
    if not vectors:
        return [], []
    dim = len(vectors[0])
    centroids = list(vectors[:k])
    if len(centroids) < k:
        centroids.extend([vectors[0]] * (k - len(centroids)))
    assignments = [0] * len(vectors)
    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for idx, vec in enumerate(vectors):
            sims = [cosine_sim(vec, centroids[i]) for i in range(k)]
            best_idx = max(range(k), key=sims.__getitem__)
            assignments[idx] = best_idx
            clusters[best_idx].append(vec)
        new_centroids = []
        for i in range(k):
            if clusters[i]:
                new_centroids.append(mean_vector(clusters[i], dim))
            else:
                new_centroids.append(centroids[i])
        delta = sum(
            sum(abs(a - b) for a, b in zip(new_centroids[i], centroids[i]))
            for i in range(k)
        )
        centroids = new_centroids
        if delta < 1e-6:
            break
    return centroids, assignments


def assign_categories(categories, features, ood_threshold=0.25, unsupervised=False):
    all_texts = categories + features
    vectors = build_tfidf_vectors(all_texts)
    cat_vecs = vectors[: len(categories)]
    feat_vecs = vectors[len(categories) :]
    if not categories:
        return [(feature, OOD_LABEL, 0.0) for feature in features]

    dim = len(cat_vecs[0]) if cat_vecs else 0
    if unsupervised:
        centroids, cluster_ids = kmeans(feat_vecs, len(categories))
        centroid_to_category = []
        for cvec in centroids:
            sims = [cosine_sim(cvec, ccat) for ccat in cat_vecs]
            best_idx = max(range(len(sims)), key=sims.__getitem__)
            centroid_to_category.append(best_idx)
        ood_centroid = mean_vector(feat_vecs, dim)
        assignments = []
        for feature, fvec, cluster_id in zip(features, feat_vecs, cluster_ids):
            centroid_vec = centroids[cluster_id]
            best_sim = cosine_sim(fvec, centroid_vec) if dim else 0.0
            ood_sim = cosine_sim(fvec, ood_centroid) if dim else 0.0
            if best_sim < ood_threshold or ood_sim > best_sim:
                assignments.append((feature, OOD_LABEL, ood_sim))
            else:
                cat_idx = centroid_to_category[cluster_id]
                assignments.append((feature, categories[cat_idx], best_sim))
        return assignments

    ood_centroid = mean_vector(feat_vecs, dim)
    assignments = []
    for _ in range(2):
        assignments = []
        ood_vectors = []
        for feature, fvec in zip(features, feat_vecs):
            sims = [cosine_sim(fvec, cvec) for cvec in cat_vecs]
            best_idx = max(range(len(sims)), key=sims.__getitem__)
            best_sim = sims[best_idx]
            ood_sim = cosine_sim(fvec, ood_centroid) if dim else 0.0

            if best_sim < ood_threshold or ood_sim > best_sim:
                assignments.append((feature, OOD_LABEL, ood_sim))
                ood_vectors.append(fvec)
            else:
                assignments.append((feature, categories[best_idx], best_sim))
        if ood_vectors:
            ood_centroid = mean_vector(ood_vectors, dim)
    return assignments


def apply_overrides(assignments, overrides):
    if not overrides:
        return assignments
    updated = []
    for feature, category, score in assignments:
        if feature in overrides:
            updated.append((feature, overrides[feature], score))
        else:
            updated.append((feature, category, score))
    return updated


def compute_coverage(categories, assignments):
    assigned = {category for _, category, _ in assignments if category != OOD_LABEL}
    hit = sum(1 for category in categories if category in assigned)
    if not categories:
        return 0.0
    return hit / len(categories)


def compute_novelty(assignments):
    if not assignments:
        return 0.0
    ood_count = sum(1 for _, category, _ in assignments if category == OOD_LABEL)
    return ood_count / len(assignments)


def load_tasks(path):
    if not path:
        return []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix == ".jsonl":
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "tasks" in data:
            return data["tasks"]
        return data
    raise ValueError(f"Unsupported tasks file extension: {path.suffix}")


def compute_accuracy(tasks, voting_key=None, success_key=None):
    if not tasks:
        return None
    voting_keys = [voting_key] if voting_key else [
        "voting_passed", "has_algo", "found", "present"
    ]
    success_keys = [success_key] if success_key else [
        "success", "passed", "tests_passed"
    ]
    voting_vals = []
    success_vals = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        for key in voting_keys:
            if key in task:
                voting_vals.append(bool(task[key]))
                break
        for key in success_keys:
            if key in task:
                success_vals.append(bool(task[key]))
                break
    voting_rate = sum(voting_vals) / len(voting_vals) if voting_vals else None
    success_rate = sum(success_vals) / len(success_vals) if success_vals else None
    return {
        "voting_rate": voting_rate,
        "success_rate": success_rate,
        "voting_total": len(voting_vals),
        "success_total": len(success_vals),
    }


def docstring_line_spans(tree):
    spans = set()
    nodes = [tree]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, (ast.Str, ast.Constant)):
                    if isinstance(first.value, ast.Constant) and not isinstance(first.value.value, str):
                        continue
                    start = getattr(first, "lineno", None)
                    end = getattr(first, "end_lineno", None) or start
                    if start and end:
                        spans.update(range(start, end + 1))
    return spans


def normalized_code_stats(repo_dir, exclude_dirs):
    repo_dir = Path(repo_dir)
    py_files = []
    for path in repo_dir.rglob("*.py"):
        parts = {p.lower() for p in path.parts}
        if any(excl in parts for excl in exclude_dirs):
            continue
        py_files.append(path)

    file_count = 0
    token_count = 0
    loc_lines = set()

    for path in py_files:
        try:
            source = path.read_text(encoding="utf-8")
        except Exception:
            continue
        file_count += 1
        try:
            tree = ast.parse(source)
            doc_lines = docstring_line_spans(tree)
        except Exception:
            doc_lines = set()

        tokens = []
        try:
            for tok in tokenize(BytesIO(source.encode("utf-8")).readline):
                if tok.type in (COMMENT, NL, NEWLINE, INDENT, DEDENT, ENCODING):
                    continue
                if tok.type == STRING and tok.start[0] in doc_lines:
                    continue
                tokens.append(tok)
        except Exception:
            continue

        token_count += len(tokens)
        for tok in tokens:
            loc_lines.add((path, tok.start[0]))

    normalized_loc = len(loc_lines)
    return {
        "file_count": file_count,
        "normalized_loc": normalized_loc,
        "token_count": token_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Repo-level evaluation script.")
    parser.add_argument("--categories", required=True, help="Path to reference categories.")
    parser.add_argument("--features", required=True, help="Path to generated features.")
    parser.add_argument("--repo_dir", required=True, help="Target repository directory.")
    parser.add_argument("--tasks_file", default="", help="Path to task results JSON/JSONL.")
    parser.add_argument("--override_map", default="", help="Optional JSON mapping for LLM overrides.")
    parser.add_argument("--text_field", default="", help="Field name for text in JSON items.")
    parser.add_argument("--ood_threshold", type=float, default=0.25)
    parser.add_argument(
        "--unsupervised",
        action="store_true",
        help="Use unsupervised k-means on features instead of fixed centroids.",
    )
    parser.add_argument("--exclude_dirs", default="tests,test,examples,example,benchmarks,benchmark,docs,doc,build,dist,.git,__pycache__,.venv,venv")
    parser.add_argument("--voting_key", default="")
    parser.add_argument("--success_key", default="")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    args = parser.parse_args()

    categories = load_text_items(args.categories, text_field=args.text_field or None)
    features = load_text_items(args.features, text_field=args.text_field or None)

    assignments = assign_categories(
        categories,
        features,
        ood_threshold=args.ood_threshold,
        unsupervised=args.unsupervised,
    )
    overrides = None
    if args.override_map:
        with open(args.override_map, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        assignments = apply_overrides(assignments, overrides)

    coverage = compute_coverage(categories, assignments)
    novelty = compute_novelty(assignments)

    tasks = load_tasks(args.tasks_file) if args.tasks_file else []
    accuracy = compute_accuracy(
        tasks,
        voting_key=args.voting_key or None,
        success_key=args.success_key or None,
    )

    exclude_dirs = {d.strip().lower() for d in args.exclude_dirs.split(",") if d.strip()}
    code_stats = normalized_code_stats(args.repo_dir, exclude_dirs)

    result = {
        "coverage": coverage,
        "novelty": novelty,
        "accuracy": accuracy,
        "code_stats": code_stats,
        "assignments": [
            {"feature": feature, "category": category, "score": score}
            for feature, category, score in assignments
        ],
    }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=True, indent=2)
    else:
        print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
