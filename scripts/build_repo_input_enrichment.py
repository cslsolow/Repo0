#!/usr/bin/env python3
"""Apply optional text supplements to repo_input README.req files.

The artifact keeps repo_input as raw requirement inputs only. This helper updates
README.req text in place and intentionally does not create manifests,
readme_output directories, or generated requirement JSON files.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "repo_input"

README_SUPPLEMENTS = {
    "django": [
        "The framework should explicitly support configuration validation, migration correctness checks, and compatibility diagnostics that help detect schema drift, deprecated configuration patterns, and environment-specific runtime mismatches before deployment.",
        "The system should expose developer-facing inspection and reporting surfaces for settings, URL/view/model metadata, and execution diagnostics so documentation, runtime behavior, and public API contracts remain aligned.",
        "The platform should include high-level support for test-oriented developer utilities, introspection hooks, and consistency checks that make framework semantics observable without prescribing implementation structure.",
    ],
    "pandas": [
        "In addition to broad analytical coverage, pandas should make dtype semantics, missing-value handling, index and label preservation, and shape-changing operations explicit, user-visible requirements across grouped, windowed, and reshaping workflows.",
        "It should also provide robust I/O and interchange behavior across common tabular storage formats and Python data containers, with practical guarantees around serialization, round-tripping, and metadata preservation.",
        "The library should further expose diagnostics and developer-facing validation surfaces that keep edge-case dataframe behavior inspectable and predictable under heterogeneous inputs.",
    ],
    "requests": [
        "The library should explicitly cover session lifecycle semantics, adapter layering, retries, streaming transfers, proxy and certificate configuration, and environment-sensitive runtime behavior without exposing implementation recipes.",
        "The system should provide high-level diagnostics and debugging surfaces for request preparation, transport decisions, compatibility checks, and optional environment integrations so failure modes remain observable and reproducible.",
        "The project should include developer- and test-oriented capability expectations around cache reuse, request/response validation, and interoperability with surrounding HTTP tooling and deployment environments.",
    ],
    "scikit-learn": [
        "The framework should explicitly include environment diagnostics, dependency and version consistency checks, optional dependency detection, and reproducible runtime configuration surfaces so users can validate installation and execution context.",
        "The library should make calibration, uncertainty diagnostics, probabilistic evaluation, statistical estimation utilities, and metadata-rich reporting first-class high-level requirements rather than leaving them implied by broader algorithm families.",
        "The project should also cover benchmark-sensitive ecosystem features such as dataset and test utilities, persistence and export interoperability, developer-facing validation helpers, and runtime/parallel configuration observability.",
    ],
    "statsmodels": [
        "The package should explicitly require rich statistical diagnostics, result object fidelity, inferential summaries, and validation/reporting surfaces that preserve modeling assumptions, compatibility metadata, and reproducibility context.",
        "The framework should make missing-data behavior, weighting semantics, constraint handling, numerical stability diagnostics, and data-interface interoperability explicit high-level obligations across model families.",
        "The system should also support persistence, export, and environment-aware compatibility checks for statistical artifacts without constraining internal implementation choices.",
    ],
    "sympy": [
        "The system should explicitly require stable symbolic object semantics, canonicalization behavior, transformation and simplification consistency, assumptions-aware reasoning surfaces, and diagnostics for ambiguous symbolic states.",
        "The library should cover high-level obligations for printing, serialization, interchange, and bridges to numerical execution environments while preserving symbolic meaning and compatibility guarantees.",
        "The project should also include developer-facing validation, documentation consistency, and test utility expectations so symbolic behavior remains observable, reproducible, and inspectable across workflows.",
    ],
}


def enrich_readme(repo: str, original_text: str) -> str:
    supplement = " ".join(line.strip() for line in README_SUPPLEMENTS[repo] if line.strip())
    if not supplement or supplement in original_text:
        return original_text
    return f"{original_text.rstrip()}\n\n{supplement}\n"


def build_repo(repo: str) -> bool:
    readme_path = OUTPUT_ROOT / repo / "README.req"
    if not readme_path.exists():
        raise FileNotFoundError(f"Missing README.req for {repo}: {readme_path}")
    original = readme_path.read_text(encoding="utf-8")
    enriched = enrich_readme(repo, original)
    if enriched == original:
        return False
    readme_path.write_text(enriched, encoding="utf-8")
    return True


def main() -> None:
    updated = [repo for repo in README_SUPPLEMENTS if build_repo(repo)]
    print(f"Updated {len(updated)} README.req files under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
