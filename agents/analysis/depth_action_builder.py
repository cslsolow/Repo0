"""Build depth-stage action candidates from generated code capabilities."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


_IDENTIFIER_WORD_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+")
_QWEN3_EMBED_LOCAL_PATH = os.environ.get("REPO0_QWEN3_EMBED_LOCAL_PATH", "").strip()
_QWEN3_EMBED_HF_ID = "Qwen/Qwen3-Embedding-0.6B"


def _humanize_identifier(text: str) -> str:
    cleaned = str(text or "").strip().replace("-", "_").replace(".", "_")
    if not cleaned:
        return ""
    parts: List[str] = []
    for chunk in cleaned.split("_"):
        if not chunk:
            continue
        parts.extend(_IDENTIFIER_WORD_RE.findall(chunk))
    return " ".join(part.lower() for part in parts if part).strip()


def _normalize_symbol(text: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", str(text or "").lower())


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


@dataclass
class CapabilitySignature:
    capability_id: str
    name: str
    description: str
    file_path: str
    module: str
    qualname: str
    parent_task: str
    parent_component: str
    calls_out: Set[str]
    called_by: Set[str]

    def name_tokens(self) -> Set[str]:
        return {token for token in _humanize_identifier(self.name).split() if len(token) >= 3}

    def desc_tokens(self) -> Set[str]:
        return {token for token in _humanize_identifier(self.description).split() if len(token) >= 3}

    def call_context(self) -> Set[str]:
        outgoing = {_normalize_symbol(item) for item in self.calls_out if item}
        incoming = {_normalize_symbol(item) for item in self.called_by if item}
        return {item for item in outgoing | incoming if item}


class _CallCollector(ast.NodeVisitor):
    """Collect call names inside a function/method body."""

    def __init__(self) -> None:
        self.calls: Set[str] = set()

    def visit_Call(self, node: ast.Call) -> Any:
        call_name = self._resolve_call_name(node.func)
        if call_name:
            self.calls.add(call_name)
        self.generic_visit(node)

    def _resolve_call_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            # Prefer the terminal attribute to keep cross-file matching stable.
            return node.attr
        return ""


class _QwenEmbedding:
    """Embedding helper aligned with RPG coverage evaluator."""

    def __init__(self, model_id: Optional[str] = None, batch_size: int = 64) -> None:
        self.model_id = model_id
        self.batch_size = max(1, int(batch_size))
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._available = None
        self.loaded_model_id = ""

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self._try_load()
        return bool(self._available)

    def _try_load(self) -> bool:
        from transformers import AutoModel, AutoTokenizer
        import torch

        # Fixed model policy:
        # 1) Prefer REPO0_QWEN3_EMBED_LOCAL_PATH when provided.
        # 2) Fallback to the HuggingFace model id.
        if self.model_id:
            candidates = [self.model_id]
        else:
            candidates = [c for c in (_QWEN3_EMBED_LOCAL_PATH, _QWEN3_EMBED_HF_ID) if c]

        last_exc: Exception | None = None
        for candidate in candidates:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(candidate, use_fast=False)
                self._model = AutoModel.from_pretrained(candidate, trust_remote_code=True)
                self._model.eval()
                self._torch = torch
                self.loaded_model_id = str(candidate)
                logging.info("DepthActionBuilder embedding model loaded: %s", candidate)
                return True
            except Exception as exc:
                last_exc = exc

        logging.warning(
            "DepthActionBuilder failed to load fixed Qwen3 embedding model candidates %s: %s. "
            "Falling back to lexical similarity.",
            candidates,
            last_exc,
        )
        self._model = None
        self._tokenizer = None
        self._torch = None
        self.loaded_model_id = ""
        return False

    def encode(self, texts: Sequence[str]) -> Optional[np.ndarray]:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if not self.available or self._model is None or self._tokenizer is None or self._torch is None:
            return None

        chunks: List[np.ndarray] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                tokenized = self._tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                outputs = self._model(**tokenized)
                embeddings = outputs.last_hidden_state[:, 0].float()
                embeddings = self._torch.nn.functional.normalize(embeddings, dim=1).cpu().numpy()
                chunks.append(embeddings.astype(np.float32))

        return np.vstack(chunks) if chunks else np.zeros((0, 1), dtype=np.float32)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class DepthActionBuilder:
    """Generate merge/revise/split candidates for breadth -> depth transition."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        name_sim_threshold: float = 0.85,
        desc_sim_threshold: float = 0.75,
        call_overlap_threshold: float = 0.4,
        max_pairs: int = 50000,
        split_min_fanout: int = 8,
    ) -> None:
        self.model_id = model_id
        self.name_sim_threshold = float(name_sim_threshold)
        self.desc_sim_threshold = float(desc_sim_threshold)
        self.call_overlap_threshold = float(call_overlap_threshold)
        self.max_pairs = max(1, int(max_pairs))
        self.split_min_fanout = max(3, int(split_min_fanout))
        self.embedding = _QwenEmbedding(model_id=model_id)

    def build_from_generated_entries(
        self,
        generated_root: str | Path,
        generated_entries: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        root = Path(generated_root).resolve()
        file_records = self._collect_python_file_records(root, generated_entries or [])
        capabilities = self._extract_capabilities(root, file_records)
        return self._build_actions(root, capabilities)

    def build_from_file_list(
        self,
        generated_root: str | Path,
        file_paths: Sequence[str | Path],
    ) -> Dict[str, Any]:
        root = Path(generated_root).resolve()
        file_records = []
        for item in file_paths:
            path = Path(item).resolve()
            if path.suffix == ".py" and path.exists():
                file_records.append({"path": path, "metadata": {}})
        capabilities = self._extract_capabilities(root, file_records)
        return self._build_actions(root, capabilities)

    def _collect_python_file_records(
        self,
        generated_root: Path,
        generated_entries: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        by_path: Dict[str, Dict[str, Any]] = {}

        for entry in generated_entries:
            if not isinstance(entry, dict):
                continue
            files = entry.get("files", {})
            if not isinstance(files, dict):
                continue
            code_path_raw = files.get("code")
            if not code_path_raw:
                continue
            code_path = Path(str(code_path_raw))
            if not code_path.is_absolute():
                code_path = (generated_root / code_path).resolve()
            if not code_path.exists() or code_path.suffix != ".py":
                continue
            by_path[str(code_path)] = {
                "path": code_path,
                "metadata": {
                    "parent_task": str(entry.get("parent_task") or entry.get("task") or "").strip(),
                    "parent_component": str(entry.get("component") or "").strip(),
                },
            }

        if not by_path and generated_root.exists():
            for path in generated_root.rglob("*.py"):
                if not path.is_file():
                    continue
                by_path[str(path.resolve())] = {"path": path.resolve(), "metadata": {}}

        records = sorted(by_path.values(), key=lambda item: str(item["path"]))
        return records

    def _extract_capabilities(
        self,
        generated_root: Path,
        file_records: Sequence[Dict[str, Any]],
    ) -> List[CapabilitySignature]:
        capabilities: List[CapabilitySignature] = []
        id_to_index: Dict[str, int] = {}
        name_to_ids: Dict[str, Set[str]] = {}

        for record in file_records:
            path = Path(record["path"])
            metadata = dict(record.get("metadata") or {})
            try:
                source = path.read_text(encoding="utf-8")
            except Exception as exc:
                logging.warning("DepthActionBuilder failed to read %s: %s", path, exc)
                continue

            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError as exc:
                logging.warning("DepthActionBuilder syntax parse skip %s: %s", path, exc)
                continue

            try:
                rel_path = path.relative_to(generated_root).as_posix()
            except ValueError:
                rel_path = path.as_posix()
            module_name = rel_path[:-3].replace("/", ".") if rel_path.endswith(".py") else rel_path.replace("/", ".")

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    cap = self._build_capability(node, module_name, rel_path, metadata, class_name="")
                    if cap:
                        id_to_index[cap.capability_id] = len(capabilities)
                        capabilities.append(cap)
                        name_to_ids.setdefault(_normalize_symbol(cap.name), set()).add(cap.capability_id)
                elif isinstance(node, ast.ClassDef):
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            cap = self._build_capability(
                                child,
                                module_name,
                                rel_path,
                                metadata,
                                class_name=node.name,
                            )
                            if cap:
                                id_to_index[cap.capability_id] = len(capabilities)
                                capabilities.append(cap)
                                name_to_ids.setdefault(_normalize_symbol(cap.name), set()).add(cap.capability_id)

        # Build project-level call graph edges based on function/method name resolution.
        for cap in capabilities:
            caller_id = cap.capability_id
            caller_idx = id_to_index[caller_id]
            outgoing_targets: Set[str] = set()
            for call in cap.calls_out:
                key = _normalize_symbol(call)
                if not key:
                    continue
                for target_id in name_to_ids.get(key, set()):
                    if target_id != caller_id:
                        outgoing_targets.add(target_id)
            # Persist neighbor ids into calls_out for overlap scoring.
            if outgoing_targets:
                capabilities[caller_idx].calls_out = set(outgoing_targets)

        # Reverse edges.
        for cap in capabilities:
            for target_id in cap.calls_out:
                target_idx = id_to_index.get(target_id)
                if target_idx is not None:
                    capabilities[target_idx].called_by.add(cap.capability_id)

        return capabilities

    def _build_capability(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        module_name: str,
        rel_path: str,
        metadata: Dict[str, Any],
        class_name: str,
    ) -> Optional[CapabilitySignature]:
        func_name = str(node.name).strip()
        if not func_name:
            return None
        if func_name.startswith("__") and func_name.endswith("__"):
            # Keep magic methods out of merge candidate generation by default.
            return None

        collector = _CallCollector()
        collector.visit(node)

        doc = ast.get_docstring(node, clean=True) or ""
        first_line = doc.splitlines()[0].strip() if doc else ""
        description = first_line or _humanize_identifier(func_name) or func_name

        qualname = f"{class_name}.{func_name}" if class_name else func_name
        capability_id = f"{module_name}::{qualname}"
        return CapabilitySignature(
            capability_id=capability_id,
            name=func_name,
            description=description,
            file_path=rel_path,
            module=module_name,
            qualname=qualname,
            parent_task=str(metadata.get("parent_task") or "").strip(),
            parent_component=str(metadata.get("parent_component") or "").strip(),
            calls_out=set(collector.calls),
            called_by=set(),
        )

    def _build_actions(self, generated_root: Path, capabilities: List[CapabilitySignature]) -> Dict[str, Any]:
        if len(capabilities) < 2:
            return {
                "builder": "DepthActionBuilder",
                "version": "1.0",
                "generated_root": str(generated_root),
                "embedding_model": self.embedding.loaded_model_id or (self.model_id or _QWEN3_EMBED_HF_ID),
                "embedding_backend": "qwen3-0.6b" if self.embedding.available else "lexical_fallback",
                "thresholds": self._thresholds(),
                "stats": {
                    "capabilities": len(capabilities),
                    "candidate_pairs": 0,
                    "merge_groups": 0,
                    "revise_candidates": 0,
                    "split_candidates": 0,
                },
                "actions": [],
                "note": "Not enough capabilities to generate depth candidates.",
            }

        name_texts = [_humanize_identifier(cap.name) or cap.name for cap in capabilities]
        desc_texts = [cap.description or _humanize_identifier(cap.name) or cap.name for cap in capabilities]
        name_embeddings = self.embedding.encode(name_texts)
        desc_embeddings = self.embedding.encode(desc_texts)

        candidate_pairs = self._candidate_pairs(capabilities)
        if len(candidate_pairs) > self.max_pairs:
            candidate_pairs = candidate_pairs[: self.max_pairs]

        merge_pairs: List[Tuple[int, int, float, float, float]] = []
        revise_pairs: List[Tuple[int, int, float, float, float]] = []
        for i, j in candidate_pairs:
            name_sim = self._similarity(i, j, name_embeddings, name_texts)
            if name_sim < self.name_sim_threshold:
                continue
            desc_sim = self._similarity(i, j, desc_embeddings, desc_texts)
            if desc_sim < self.desc_sim_threshold:
                continue
            call_overlap = _jaccard(capabilities[i].call_context(), capabilities[j].call_context())
            if call_overlap >= self.call_overlap_threshold:
                merge_pairs.append((i, j, name_sim, desc_sim, call_overlap))
            else:
                revise_pairs.append((i, j, name_sim, desc_sim, call_overlap))

        uf = _UnionFind(len(capabilities))
        for i, j, _, _, _ in merge_pairs:
            uf.union(i, j)

        grouped: Dict[int, List[int]] = {}
        for idx in range(len(capabilities)):
            root = uf.find(idx)
            grouped.setdefault(root, []).append(idx)

        actions: List[Dict[str, Any]] = []
        for members in grouped.values():
            if len(members) < 2:
                continue
            member_set = set(members)
            pair_scores = [
                (name_sim, desc_sim, call_overlap)
                for i, j, name_sim, desc_sim, call_overlap in merge_pairs
                if i in member_set and j in member_set
            ]
            if not pair_scores:
                continue
            name_avg = float(np.mean([item[0] for item in pair_scores]))
            desc_avg = float(np.mean([item[1] for item in pair_scores]))
            call_avg = float(np.mean([item[2] for item in pair_scores]))

            targets = sorted(capabilities[idx].capability_id for idx in members)
            actions.append(
                {
                    "type": "merge",
                    "targets": targets,
                    "scores": {
                        "name_sim_avg": round(name_avg, 4),
                        "desc_sim_avg": round(desc_avg, 4),
                        "call_overlap_avg": round(call_avg, 4),
                    },
                    "reason": "High name/description similarity and call-graph overlap.",
                }
            )

        for i, j, name_sim, desc_sim, call_overlap in sorted(
            revise_pairs,
            key=lambda item: (item[2] + item[3] - item[4]),
            reverse=True,
        )[:200]:
            actions.append(
                {
                    "type": "revise",
                    "target": capabilities[i].capability_id,
                    "reference": capabilities[j].capability_id,
                    "scores": {
                        "name_sim": round(name_sim, 4),
                        "desc_sim": round(desc_sim, 4),
                        "call_overlap": round(call_overlap, 4),
                    },
                    "reason": "Semantically similar but call-graph overlap is below merge threshold.",
                }
            )

        for idx, cap in enumerate(capabilities):
            fanout = len(cap.calls_out)
            if fanout < self.split_min_fanout:
                continue
            call_prefixes = {
                _normalize_symbol(item).split("_", 1)[0]
                for item in cap.calls_out
                if _normalize_symbol(item)
            }
            if len(call_prefixes) < 3:
                continue
            actions.append(
                {
                    "type": "split",
                    "target": cap.capability_id,
                    "metrics": {
                        "fanout": fanout,
                        "call_prefix_groups": len(call_prefixes),
                    },
                    "reason": "High fan-out with multiple call groups suggests mixed responsibilities.",
                }
            )

        return {
            "builder": "DepthActionBuilder",
            "version": "1.0",
            "generated_root": str(generated_root),
            "embedding_model": self.embedding.loaded_model_id or (self.model_id or _QWEN3_EMBED_HF_ID),
            "embedding_backend": "qwen3-0.6b" if self.embedding.available else "lexical_fallback",
            "thresholds": self._thresholds(),
            "stats": {
                "capabilities": len(capabilities),
                "candidate_pairs": len(candidate_pairs),
                "merge_pairs": len(merge_pairs),
                "revise_pairs": len(revise_pairs),
                "merge_groups": sum(1 for action in actions if action.get("type") == "merge"),
                "revise_candidates": sum(1 for action in actions if action.get("type") == "revise"),
                "split_candidates": sum(1 for action in actions if action.get("type") == "split"),
            },
            "actions": actions,
        }

    def _candidate_pairs(self, capabilities: Sequence[CapabilitySignature]) -> List[Tuple[int, int]]:
        token_buckets: Dict[str, List[int]] = {}
        tokens_by_index: List[Set[str]] = []
        for idx, cap in enumerate(capabilities):
            tokens = cap.name_tokens()
            tokens_by_index.append(tokens)
            for token in tokens:
                token_buckets.setdefault(token, []).append(idx)

        pairs: Set[Tuple[int, int]] = set()
        for i, tokens in enumerate(tokens_by_index):
            candidate_indices: Set[int] = set()
            for token in tokens:
                candidate_indices.update(token_buckets.get(token, []))
            for j in candidate_indices:
                if j <= i:
                    continue
                pairs.add((i, j))

        if not pairs and len(capabilities) <= 300:
            for i, j in combinations(range(len(capabilities)), 2):
                pairs.add((i, j))

        return sorted(pairs)

    def _similarity(
        self,
        i: int,
        j: int,
        embeddings: Optional[np.ndarray],
        raw_texts: Sequence[str],
    ) -> float:
        if embeddings is not None and embeddings.size:
            return float(np.dot(embeddings[i], embeddings[j]))
        # lexical fallback
        left = set(_humanize_identifier(raw_texts[i]).split())
        right = set(_humanize_identifier(raw_texts[j]).split())
        return _jaccard(left, right)

    def _thresholds(self) -> Dict[str, float]:
        return {
            "name_sim": self.name_sim_threshold,
            "desc_sim": self.desc_sim_threshold,
            "call_overlap": self.call_overlap_threshold,
        }


__all__ = ["DepthActionBuilder", "CapabilitySignature"]
