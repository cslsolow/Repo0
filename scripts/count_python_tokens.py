#!/usr/bin/env python3
"""Count tokenizer tokens for Python files under a directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count tokenizer token usage for *.py files in a directory."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory to scan recursively.")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Tokenizer model name for tiktoken (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="",
        help="Explicit tiktoken encoding (e.g., cl100k_base). Overrides --model if provided.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON output file path.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=50,
        help="Maximum number of per-file rows to print (sorted by token count desc).",
    )
    return parser.parse_args()


def build_encoder(model: str, encoding: str):
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "tiktoken is required. Install with: pip install tiktoken"
        ) from exc

    if encoding.strip():
        return tiktoken.get_encoding(encoding.strip()), f"encoding:{encoding.strip()}"

    try:
        enc = tiktoken.encoding_for_model(model)
        return enc, f"model:{model}"
    except KeyError:
        # Unknown model in current tiktoken version, fallback to cl100k_base.
        fallback = "cl100k_base"
        return tiktoken.get_encoding(fallback), f"fallback:{fallback} (unknown model: {model})"


def count_tokens_for_files(input_dir: Path, encoder) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    py_files = sorted(path for path in input_dir.rglob("*.py") if path.is_file())
    rows: List[Dict[str, Any]] = []
    total_tokens = 0
    total_chars = 0
    total_bytes = 0

    for path in py_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        token_count = len(encoder.encode(text, disallowed_special=()))
        char_count = len(text)
        byte_count = len(text.encode("utf-8"))

        total_tokens += token_count
        total_chars += char_count
        total_bytes += byte_count

        rows.append(
            {
                "file": str(path),
                "tokens": token_count,
                "chars": char_count,
                "bytes": byte_count,
            }
        )

    summary = {
        "python_files": len(py_files),
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "total_bytes": total_bytes,
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    encoder, tokenizer_name = build_encoder(args.model, args.encoding)
    rows, summary = count_tokens_for_files(input_dir, encoder)
    sorted_rows = sorted(rows, key=lambda x: int(x["tokens"]), reverse=True)

    print("=== Python Token Count Summary ===")
    print(f"Input dir: {input_dir}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Python files: {summary['python_files']}")
    print(f"Total tokens: {summary['total_tokens']}")
    print(f"Total chars: {summary['total_chars']}")
    print(f"Total bytes: {summary['total_bytes']}")

    if sorted_rows:
        print("\nTop files by token count:")
        for idx, row in enumerate(sorted_rows[: max(0, int(args.max_print))], start=1):
            print(
                f"{idx}. {row['file']} | tokens={row['tokens']}, chars={row['chars']}, bytes={row['bytes']}"
            )
        if len(sorted_rows) > args.max_print:
            print(f"... {len(sorted_rows) - args.max_print} more files omitted")

    if args.json_output:
        output_path = args.json_output.resolve()
        if output_path.exists() and output_path.is_dir():
            output_path = output_path / "python_token_counts.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_dir": str(input_dir),
            "tokenizer": tokenizer_name,
            "summary": summary,
            "files": sorted_rows,
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON report: {output_path}")


if __name__ == "__main__":
    main()
