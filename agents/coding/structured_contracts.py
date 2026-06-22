"""AST-level structured state contract analysis for generated Python modules."""

from __future__ import annotations

import ast
from typing import Dict, List, Set


DICT_METHODS = {
    "clear", "copy", "get", "items", "keys", "pop", "popitem",
    "setdefault", "update", "values",
}
LIST_METHODS = {
    "append", "clear", "copy", "count", "extend", "index", "insert",
    "pop", "remove", "reverse", "sort",
}
SET_METHODS = {
    "add", "clear", "copy", "difference", "difference_update", "discard",
    "intersection", "intersection_update", "isdisjoint", "issubset",
    "issuperset", "pop", "remove", "symmetric_difference",
    "symmetric_difference_update", "union", "update",
}


def _infer_expr_kind(node: ast.AST | None) -> str:
    if node is None:
        return "unknown"
    if isinstance(node, ast.Dict):
        return "dict"
    if isinstance(node, ast.List):
        return "list"
    if isinstance(node, ast.Set):
        return "set"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if isinstance(node, ast.Constant) and node.value is None:
        return "none"
    if isinstance(node, ast.Call):
        func = node.func
        func_name = ""
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr
        lowered = func_name.lower()
        if lowered in {"dict", "defaultdict", "ordereddict", "counter"}:
            return "dict"
        if lowered == "list":
            return "list"
        if lowered == "set":
            return "set"
        return "opaque_object"
    if isinstance(node, ast.Name):
        return "name_ref"
    if isinstance(node, ast.Attribute):
        return "attr_ref"
    return "unknown"


class _StructuredStateVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.attr_kinds: Dict[str, Set[str]] = {}
        self.attr_methods: Dict[str, Set[str]] = {}

    def _record_attr_assignment(self, attr_name: str, value: ast.AST | None) -> None:
        if not attr_name:
            return
        self.attr_kinds.setdefault(attr_name, set()).add(_infer_expr_kind(value))

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                self._record_attr_assignment(target.attr, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        target = node.target
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            self._record_attr_assignment(target.attr, node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "self"
        ):
            self.attr_methods.setdefault(func.value.attr, set()).add(func.attr)
        self.generic_visit(node)


def find_structured_contract_issues(code: str, rel_path: str = "") -> List[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    visitor = _StructuredStateVisitor()
    visitor.visit(tree)
    issues: List[str] = []
    prefix = f"{rel_path}: " if rel_path else ""

    for attr_name in sorted(set(visitor.attr_kinds) | set(visitor.attr_methods)):
        kinds = visitor.attr_kinds.get(attr_name, set())
        methods = visitor.attr_methods.get(attr_name, set())
        dict_methods = sorted(methods & DICT_METHODS)
        list_methods = sorted(methods & LIST_METHODS)
        set_methods = sorted(methods & SET_METHODS)
        set_only_methods = sorted(m for m in set_methods if m not in DICT_METHODS)
        dict_only_methods = sorted(m for m in dict_methods if m not in SET_METHODS)

        if "opaque_object" in kinds and dict_methods:
            issues.append(
                f"{prefix}structured contract mismatch for self.{attr_name}: initialized as opaque object but used with dict-like methods {', '.join(dict_methods)}"
            )
        if "opaque_object" in kinds and list_methods:
            issues.append(
                f"{prefix}structured contract mismatch for self.{attr_name}: initialized as opaque object but used with list-like methods {', '.join(list_methods)}"
            )
        if "opaque_object" in kinds and set_only_methods:
            issues.append(
                f"{prefix}structured contract mismatch for self.{attr_name}: initialized as opaque object but used with set-like methods {', '.join(set_only_methods)}"
            )
        if "dict" in kinds and list_methods:
            issues.append(
                f"{prefix}structured contract mismatch for self.{attr_name}: initialized as dict but used with list-like methods {', '.join(list_methods)}"
            )
        if "dict" in kinds and set_only_methods:
            issues.append(
                f"{prefix}structured contract mismatch for self.{attr_name}: initialized as dict but used with set-like methods {', '.join(set_only_methods)}"
            )
        if "list" in kinds and dict_only_methods:
            issues.append(
                f"{prefix}structured contract mismatch for self.{attr_name}: initialized as list but used with dict-like methods {', '.join(dict_only_methods)}"
            )
        if "set" in kinds and dict_only_methods:
            issues.append(
                f"{prefix}structured contract mismatch for self.{attr_name}: initialized as set but used with dict-like methods {', '.join(dict_only_methods)}"
            )
        if "dict" in kinds and "opaque_object" in kinds and dict_methods:
            issues.append(
                f"{prefix}mixed state contract for self.{attr_name}: both dict and opaque-object assignments observed before dict-like use"
            )

    return sorted(set(issues))


def extract_structured_contract_facts(code: str, rel_path: str = "") -> List[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    visitor = _StructuredStateVisitor()
    visitor.visit(tree)
    facts: List[str] = []
    prefix = f"{rel_path}: " if rel_path else ""

    for attr_name in sorted(set(visitor.attr_kinds) | set(visitor.attr_methods)):
        kinds = set(visitor.attr_kinds.get(attr_name, set()))
        methods = set(visitor.attr_methods.get(attr_name, set()))
        if not kinds and not methods:
            continue

        stable_kind = ""
        if "dict" in kinds:
            stable_kind = "dict-like"
        elif "list" in kinds:
            stable_kind = "list-like"
        elif "set" in kinds:
            stable_kind = "set-like"
        elif "opaque_object" in kinds:
            stable_kind = "opaque-object"
        elif len(kinds) == 1:
            stable_kind = next(iter(kinds))

        if not stable_kind:
            continue

        method_suffix = ""
        if methods:
            shown = ", ".join(sorted(methods)[:4])
            method_suffix = f" using methods {shown}"

        facts.append(f"{prefix}self.{attr_name} should remain {stable_kind}{method_suffix}")

    return sorted(set(facts))


__all__ = ["find_structured_contract_issues", "extract_structured_contract_facts"]
