#!/usr/bin/env python3
"""Probe which max_tokens values a configured chat model accepts.

The default "accept" mode is cheap: it asks for a one-token response while
varying the requested max_tokens value. Use "generate" mode only when you need
to confirm that a provider can actually emit long completions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_VALUES = [8196, 9000, 12000, 16000, 20000, 32768, 48000]


def _mask_key(key: str) -> str:
    if not key:
        return "<empty>"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def _parse_values(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise argparse.ArgumentTypeError("max token values must be positive")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one max token value is required")
    return sorted(set(values))


def _build_prompt(mode: str, target_tokens: int) -> list[dict[str, str]]:
    if mode == "accept":
        return [
            {"role": "system", "content": "Return only the word OK."},
            {"role": "user", "content": "OK"},
        ]
    return [
        {
            "role": "system",
            "content": (
                "You are testing output length. Output only repeated lowercase "
                "letter x tokens separated by spaces. Do not explain."
            ),
        },
        {
            "role": "user",
            "content": f"Output approximately {target_tokens} tokens.",
        },
    ]


def probe_value(
    client: OpenAI,
    *,
    model: str,
    base_url: str,
    reasoning_effort: str,
    mode: str,
    value: int,
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": _build_prompt(mode, value),
        "temperature": temperature,
        "max_tokens": value,
        "timeout": timeout,
    }
    if reasoning_effort:
        request_kwargs["reasoning_effort"] = reasoning_effort

    result: dict[str, Any] = {
        "max_tokens_requested": value,
        "accepted": False,
        "mode": mode,
        "model": model,
        "base_url": base_url,
    }
    try:
        response = client.chat.completions.create(**request_kwargs)
        choice = response.choices[0] if getattr(response, "choices", None) else None
        usage = getattr(response, "usage", None)
        text = ""
        if choice is not None:
            message = getattr(choice, "message", None)
            text = str(getattr(message, "content", "") or "")
        result.update(
            {
                "accepted": True,
                "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
                "response_chars": len(text),
                "response_preview": text[:120],
            }
        )
    except Exception as exc:
        result.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe max_tokens acceptance for the configured LLM endpoint."
    )
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5-mini"))
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("OPENAI_REASONING_EFFORT", ""),
        help="Optional reasoning_effort parameter.",
    )
    parser.add_argument(
        "--values",
        type=_parse_values,
        default=DEFAULT_VALUES,
        help="Comma-separated max_tokens values to probe.",
    )
    parser.add_argument(
        "--previous-max",
        type=int,
        default=32768,
        help="Previous largest observed per-request output token count to cover.",
    )
    parser.add_argument(
        "--mode",
        choices=["accept", "generate"],
        default="accept",
        help="accept is cheap; generate asks for long completions and costs tokens.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.api_key:
        print("Missing API key. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2
    if not args.base_url:
        print("Missing base URL. Set OPENAI_BASE_URL or pass --base-url.", file=sys.stderr)
        return 2

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    results = [
        probe_value(
            client,
            model=args.model,
            base_url=args.base_url,
            reasoning_effort=args.reasoning_effort,
            mode=args.mode,
            value=value,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        for value in args.values
    ]
    accepted_values = [r["max_tokens_requested"] for r in results if r.get("accepted")]
    max_accepted = max(accepted_values) if accepted_values else None
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "api_key": _mask_key(args.api_key),
        "mode": args.mode,
        "previous_max": args.previous_max,
        "max_accepted_requested_max_tokens": max_accepted,
        "covers_previous_max": bool(max_accepted is not None and max_accepted >= args.previous_max),
        "results": results,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if report["covers_previous_max"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
