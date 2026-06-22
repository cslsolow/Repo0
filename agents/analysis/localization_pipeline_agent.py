"""Localization agent implementing the C.1 toolset."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from agents.infra.llm_client import LLMClient
except Exception:
    LLMClient = None

logger = logging.getLogger("localization_agent")

_FEATURE_MAP_CACHE: Dict[Path, Dict[str, Any]] = {}
_MAX_INTERFACE_SPECS = 8
_MAX_INTERFACE_CHARS = 2000
_FEATURE_TOKEN_WEIGHT = 0.6


def _locate_agents_output_dir(file_path: Path) -> Optional[Path]:
    for parent in file_path.parents:
        if parent.name == "agents_output":
            return parent
    for parent in file_path.parents:
        candidate = parent / "agents_output"
        if candidate.is_dir():
            return candidate
    return None


def _load_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _build_feature_maps(agents_output_dir: Path) -> Dict[str, Any]:
    cache = _FEATURE_MAP_CACHE.get(agents_output_dir)
    if cache is not None:
        return cache

    component_map: Dict[str, List[str]] = {}
    subreq_desc: Dict[str, str] = {}
    actions_map: Dict[str, List[Dict[str, str]]] = {}
    file_component_map: Dict[str, List[str]] = {}
    suffix_component_map: Dict[str, List[str]] = {}
    suffix_markers = ["generated_code"]

    arch_path = agents_output_dir / "architectures_flattened.json"
    if not arch_path.exists():
        arch_path = agents_output_dir / "architectures.json"
    arch_data = _load_json_file(arch_path) if arch_path.exists() else None

    if isinstance(arch_data, list):
        for item in arch_data:
            architecture = item.get("architecture") if isinstance(item, dict) else None
            subreqs = []
            components = []
            if isinstance(architecture, dict):
                subreqs = architecture.get("sub_requirements", []) or []
                components = architecture.get("components", []) or []
            elif isinstance(item, dict):
                subreqs = item.get("sub_requirements", []) or []
                components = item.get("components", []) or []

            for subreq in subreqs:
                name = subreq.get("name") if isinstance(subreq, dict) else None
                if not name:
                    continue
                desc = subreq.get("description") if isinstance(subreq, dict) else ""
                subreq_desc[name] = desc

            for comp in components:
                if not isinstance(comp, dict):
                    continue
                comp_name = comp.get("name")
                if not comp_name:
                    continue
                serves = comp.get("serves_subrequirements", []) or []
                component_map.setdefault(comp_name, [])
                for sub in serves:
                    if sub not in component_map[comp_name]:
                        component_map[comp_name].append(sub)

    actions_path = agents_output_dir / "actions.json"
    actions_data = _load_json_file(actions_path) if actions_path.exists() else None
    if isinstance(actions_data, list):
        for item in actions_data:
            for action in item.get("actions", []) if isinstance(item, dict) else []:
                if not isinstance(action, dict):
                    continue
                comp = action.get("component")
                if not comp:
                    continue
                actions_map.setdefault(comp, [])
                entry = {
                    "action": action.get("action", ""),
                    "rationale": action.get("rationale", ""),
                }
                actions_map[comp].append(entry)

    generated_path = agents_output_dir / "generated_files.json"
    generated_data = _load_json_file(generated_path) if generated_path.exists() else None
    if isinstance(generated_data, list):
        for item in generated_data:
            if not isinstance(item, dict):
                continue
            comp = item.get("component")
            files = item.get("files", {})
            code_path = files.get("code") if isinstance(files, dict) else None
            if not comp or not code_path:
                continue
            resolved = str(Path(code_path))
            file_component_map.setdefault(resolved, [])
            if comp not in file_component_map[resolved]:
                file_component_map[resolved].append(comp)
            suffix = _path_suffix_after_any(resolved, suffix_markers)
            if suffix:
                suffix_component_map.setdefault(suffix, [])
                if comp not in suffix_component_map[suffix]:
                    suffix_component_map[suffix].append(comp)

    cache = {
        "component_map": component_map,
        "subreq_desc": subreq_desc,
        "actions_map": actions_map,
        "file_component_map": file_component_map,
        "suffix_component_map": suffix_component_map,
        "suffix_markers": suffix_markers,
    }
    _FEATURE_MAP_CACHE[agents_output_dir] = cache
    return cache


def _match_components(names: List[str], component_map: Dict[str, List[str]]) -> List[str]:
    matched = []
    lowered = {name.lower() for name in names if name}
    for comp in component_map.keys():
        if comp.lower() in lowered:
            matched.append(comp)
    return matched


def _path_suffix_after(path: str, marker: str) -> str:
    parts = Path(path).parts
    if marker in parts:
        idx = parts.index(marker)
        suffix_parts = parts[idx + 1 :]
        if suffix_parts:
            return str(Path(*suffix_parts))
    return ""


def _path_suffix_after_any(path: str, markers: List[str]) -> str:
    for marker in markers:
        suffix = _path_suffix_after(path, marker)
        if suffix:
            return suffix
    return ""


def _shorten_subreq_name(name: str) -> str:
    if "::" in name:
        return name.split("::", 1)[1]
    return name


def _truncate_content(content: str) -> str:
    if len(content) <= _MAX_INTERFACE_CHARS:
        return content
    return content[:_MAX_INTERFACE_CHARS].rstrip() + "\n... [truncated]"


def _feature_maps_for_interfaces(interfaces: List[InterfaceInfo]) -> Dict[str, Any]:
    if not interfaces:
        return {
            "component_map": {},
            "subreq_desc": {},
            "actions_map": {},
            "file_component_map": {},
            "suffix_component_map": {},
        }
    agents_output_dir = _locate_agents_output_dir(interfaces[0].file_path)
    if not agents_output_dir:
        return {
            "component_map": {},
            "subreq_desc": {},
            "actions_map": {},
            "file_component_map": {},
            "suffix_component_map": {},
        }
    return _build_feature_maps(agents_output_dir)


def _feature_tokens_for_interface(info: InterfaceInfo, feature_maps: Dict[str, Any]) -> List[str]:
    component_map = feature_maps.get("component_map", {})
    subreq_desc = feature_maps.get("subreq_desc", {})
    file_component_map = feature_maps.get("file_component_map", {})
    suffix_component_map = feature_maps.get("suffix_component_map", {})
    suffix_markers = feature_maps.get("suffix_markers", ["generated_code"])
    file_components = file_component_map.get(str(info.file_path), [])
    suffix = _path_suffix_after_any(str(info.file_path), suffix_markers)
    suffix_components = suffix_component_map.get(suffix, []) if suffix else []
    default_components = list(dict.fromkeys(file_components + suffix_components))
    components = _match_components([info.name, info.qualname], component_map) or default_components

    tokens: List[str] = []
    for comp in components:
        for sub in component_map.get(comp, []):
            tokens.extend(_tokenize(sub))
            tokens.extend(_tokenize(subreq_desc.get(sub, "")))
    return tokens


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".tox",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    "__pycache__",
}


@dataclass(frozen=True)
class InterfaceInfo:
    file_path: Path
    kind: str
    name: str
    qualname: str
    lineno: int
    docstring: str


def iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if any(part in DEFAULT_EXCLUDE_DIRS for part in path.parts):
            continue
        yield path


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z0-9_]+", text.lower()) if t]


def build_interface_index(repo_root: Path) -> List[InterfaceInfo]:
    interfaces: List[InterfaceInfo] = []
    for path in iter_python_files(repo_root):
        source = _read_text(path)
        if source is None:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                interfaces.append(
                    InterfaceInfo(
                        file_path=path,
                        kind="function",
                        name=node.name,
                        qualname=node.name,
                        lineno=node.lineno,
                        docstring=ast.get_docstring(node) or "",
                    )
                )
            if isinstance(node, ast.ClassDef):
                interfaces.append(
                    InterfaceInfo(
                        file_path=path,
                        kind="class",
                        name=node.name,
                        qualname=node.name,
                        lineno=node.lineno,
                        docstring=ast.get_docstring(node) or "",
                    )
                )
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef):
                        qualname = f"{node.name}.{sub.name}"
                        interfaces.append(
                            InterfaceInfo(
                                file_path=path,
                                kind="method",
                                name=sub.name,
                                qualname=qualname,
                                lineno=sub.lineno,
                                docstring=ast.get_docstring(sub) or "",
                            )
                        )
    return interfaces


def _score_interface(
    keywords: Sequence[str],
    info: InterfaceInfo,
    extra_tokens: Optional[Sequence[str]] = None,
) -> float:
    query = set(_tokenize(" ".join(keywords)))
    if not query:
        return 0.0
    cand = set(_tokenize(info.qualname) + _tokenize(info.docstring))
    if not cand:
        return 0.0
    overlap = len(query & cand) / max(len(query), 1)
    prefix = 0.0
    for token in query:
        if info.name.lower().startswith(token):
            prefix = max(prefix, len(token) / max(len(info.name), 1))
    extra_overlap = 0.0
    if extra_tokens:
        extra = set(_tokenize(" ".join(extra_tokens)))
        if extra:
            extra_overlap = len(query & extra) / max(len(query), 1)
    return overlap + prefix + (_FEATURE_TOKEN_WEIGHT * extra_overlap)


def _extract_node_source(source: str, node: ast.AST) -> str:
    lines = source.splitlines()
    if not hasattr(node, "lineno"):
        return ""
    start = max(node.lineno - 1, 0)
    end = getattr(node, "end_lineno", None)
    if end is None:
        end = start + 1
    return "\n".join(lines[start:end])


def _find_node_by_qualname(tree: ast.AST, qualname: str) -> Optional[ast.AST]:
    if "." in qualname:
        class_name, method_name = qualname.split(".", 1)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name == method_name:
                        return sub
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == qualname:
            return node
    return None


def view_file_interface_feature_map(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path)
    source = _read_text(path)
    if source is None:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    interfaces = []
    agents_output_dir = _locate_agents_output_dir(path)
    feature_maps = _build_feature_maps(agents_output_dir) if agents_output_dir else {
        "component_map": {},
        "subreq_desc": {},
        "actions_map": {},
        "file_component_map": {},
        "suffix_component_map": {},
    }
    component_map = feature_maps["component_map"]
    subreq_desc = feature_maps["subreq_desc"]
    actions_map = feature_maps["actions_map"]
    file_component_map = feature_maps["file_component_map"]
    suffix_component_map = feature_maps["suffix_component_map"]
    suffix_markers = feature_maps.get("suffix_markers", ["generated_code"])
    file_components = file_component_map.get(str(path), [])
    suffix = _path_suffix_after_any(str(path), suffix_markers)
    suffix_components = suffix_component_map.get(suffix, []) if suffix else []
    default_components = list(dict.fromkeys(file_components + suffix_components))

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node) or ""
            matches = _match_components([node.name], component_map) or default_components
            feature_map: Dict[str, str] = {}
            actions: Dict[Tuple[str, str], Dict[str, str]] = {}
            for comp in matches:
                for sub in component_map.get(comp, []):
                    feature_map.setdefault(sub, subreq_desc.get(sub, ""))
                for action in actions_map.get(comp, []):
                    key = (action.get("action", ""), action.get("rationale", ""))
                    actions[key] = action
            interfaces.append(
                {
                    "interface": node.name,
                    "type": "function",
                    "features": {
                        "docstring": docstring,
                        "components": matches,
                        "feature_map": [
                            {"name": _shorten_subreq_name(name), "description": desc}
                            for name, desc in feature_map.items()
                        ],
                        "actions": list(actions.values()),
                    },
                }
            )
        if isinstance(node, ast.ClassDef):
            docstring = ast.get_docstring(node) or ""
            matches = _match_components([node.name], component_map) or default_components
            feature_map: Dict[str, str] = {}
            actions: Dict[Tuple[str, str], Dict[str, str]] = {}
            method_names: List[str] = []
            for comp in matches:
                for sub in component_map.get(comp, []):
                    feature_map.setdefault(sub, subreq_desc.get(sub, ""))
                for action in actions_map.get(comp, []):
                    key = (action.get("action", ""), action.get("rationale", ""))
                    actions[key] = action
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    method_names.append(sub.name)
            interfaces.append(
                {
                    "interface": node.name,
                    "type": "class",
                    "features": {
                        "docstring": docstring,
                        "components": matches,
                        "feature_map": [
                            {"name": _shorten_subreq_name(name), "description": desc}
                            for name, desc in feature_map.items()
                        ],
                        "actions": list(actions.values()),
                        "methods": method_names,
                    },
                }
            )
    return interfaces


def get_interface_content(target_specs: List[str]) -> Dict[str, str]:
    results: Dict[str, str] = {}
    for idx, spec in enumerate(target_specs):
        if idx >= _MAX_INTERFACE_SPECS:
            results[spec] = "[omitted: too many interfaces requested]"
            continue
        if ":" not in spec:
            results[spec] = ""
            continue
        file_path, qualname = spec.split(":", 1)
        path = Path(file_path)
        source = _read_text(path)
        if source is None:
            results[spec] = ""
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            results[spec] = ""
            continue
        node = _find_node_by_qualname(tree, qualname)
        content = _extract_node_source(source, node) if node else ""
        results[spec] = _truncate_content(content)
    return results


def expand_leaf_node_info(
    feature_path: str,
    interfaces: Optional[List[InterfaceInfo]] = None,
) -> List[Dict[str, Any]]:
    keywords = _tokenize(feature_path)
    if interfaces is None:
        return []
    feature_maps = _feature_maps_for_interfaces(interfaces)
    ranked = sorted(
        interfaces,
        key=lambda info: _score_interface(
            keywords,
            info,
            extra_tokens=_feature_tokens_for_interface(info, feature_maps),
        ),
        reverse=True,
    )
    results = []
    for info in ranked[:10]:
        results.append(
            {
                "file_path": str(info.file_path),
                "interface": f"{info.kind}: {info.qualname}",
            }
        )
    return results


def search_interface_by_functionality(
    keywords: List[str],
    interfaces: Optional[List[InterfaceInfo]] = None,
) -> List[Dict[str, Any]]:
    if interfaces is None:
        return []
    feature_maps = _feature_maps_for_interfaces(interfaces)
    ranked = sorted(
        interfaces,
        key=lambda info: _score_interface(
            keywords,
            info,
            extra_tokens=_feature_tokens_for_interface(info, feature_maps),
        ),
        reverse=True,
    )
    results = []
    for info in ranked[:5]:
        results.append(
            {
                "file_path": str(info.file_path),
                "interface": f"{info.kind}: {info.qualname}",
            }
        )
    return results


def Terminate(result: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return result


class LocalizationAgent:
    """Agent providing localization tool behaviors for repository inspection."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.interfaces = build_interface_index(self.repo_root)

    def search_by_keywords(self, keywords: Sequence[str], top_k: int = 5) -> List[Dict[str, Any]]:
        feature_maps = _feature_maps_for_interfaces(self.interfaces)
        ranked = sorted(
            self.interfaces,
            key=lambda info: _score_interface(
                keywords,
                info,
                extra_tokens=_feature_tokens_for_interface(info, feature_maps),
            ),
            reverse=True,
        )
        results = []
        for info in ranked[:top_k]:
            results.append(
                {
                    "file_path": str(info.file_path),
                    "interface": f"{info.kind}: {info.qualname}",
                }
            )
        return results

    def expand_feature_path(self, feature_path: str, top_k: int = 10) -> List[Dict[str, Any]]:
        keywords = _tokenize(feature_path)
        return self.search_by_keywords(keywords, top_k=top_k)


class LLMToolLocalizationAgent:
    """LLM-driven localization agent that calls C.1 tools and terminates with ranked results."""

    def __init__(
        self,
        repo_root: Path,
        llm_client: Any,
        max_steps: int = 12,
        max_tokens: int = 2048,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.llm_client = llm_client
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.interfaces = build_interface_index(self.repo_root)

    def run(self, task: str) -> Dict[str, Any]:
        if not self.llm_client:
            raise ValueError("LLM client is required for LLMToolLocalizationAgent")

        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt(),
            },
            {
                "role": "user",
                "content": (
                    "Task:\n"
                    f"{task}\n\n"
                    "Instructions:\n"
                    "- Identify the most relevant functions/classes/methods that implement the task.\n"
                    "- Use the tools to inspect files and interfaces as needed.\n"
                    "- Provide a ranked result list via Terminate when confident.\n\n"
                    f"Repo root: {self.repo_root}"
                ),
            },
        ]

        steps = 0
        last_result: Optional[List[Dict[str, Any]]] = None
        tool_log: List[Dict[str, Any]] = []
        while steps < self.max_steps:
            steps += 1
            messages.append(
                {
                    "role": "user",
                    "content": f"Step {steps} of {self.max_steps}.",
                }
            )
            try:
                response = self.llm_client.call(messages, temperature=0.0, max_tokens=self.max_tokens)
            except Exception as exc:
                self._log_llm_error(exc)
                return {
                    "status": "llm_error",
                    "steps": steps,
                    "tool_log": tool_log,
                    "error": str(exc),
                }
            logger.debug("LLM raw response:\n%s", response)
            messages.append({"role": "assistant", "content": response})
            calls = self._parse_tool_calls(response)
            if calls is None:
                messages.append(
                    {
                        "role": "user",
                        "content": self._format_json_retry(response),
                    }
                )
                continue
            if not calls:
                return {
                    "status": "no_tool_call",
                    "steps": steps,
                    "tool_log": tool_log,
                    "last_message": response,
                }
            for call in calls:
                result = self._execute_call(call)
                tool_log.append({"call": call, "result": result})
                if call["name"] == "Terminate":
                    last_result = result
                    return {
                        "status": "terminated",
                        "steps": steps,
                        "tool_log": tool_log,
                        "result": last_result,
                    }
                messages.append(
                    {
                        "role": "user",
                        "content": self._format_tool_result(call, result),
                    }
                )

        return {
            "status": "max_steps_exceeded",
            "steps": steps,
            "tool_log": tool_log,
            "result": last_result,
        }

    def _build_system_prompt(self) -> str:
        return (
            "You will be given a task description. Your job is to localize the most relevant\n"
            "functions/classes/methods in the repository that implement the task, using the tools below.\n"
            "Explore, inspect, and then terminate with a ranked result list.\n"
            "\n"
            "Localization Tools\n"
            "### Interface Inspection Tools\n"
            "- 'view_file_interface_feature_map(file_path)'\n"
            "Inspects a single Python file to list the interface structures (functions, classes, methods) it\n"
            "contains, along with the feature mappings they support.\n"
            "*Usage*: Useful for quickly understanding which interfaces exist in a given file and the feature\n"
            "tags associated with them.\n"
            "*Example*:\n"
            "view_file_interface_feature_map('src/algorithms/classifier.py')\n"
            "- 'get_interface_content(target_specs)'\n"
            "Retrieves the full implementation code of a specific function, class, or method, given its fully\n"
            "qualified name (file path + entity name).\n"
            "*Usage*: Applied when a particular interface has been located and its source code needs to be\n"
            "examined in detail.\n"
            "*Example*:\n"
            "get_interface_content(['src/core/data_loader.py:DataLoader.load_data'])\n"
            "get_interface_content(['src/core/utils.py:clean_text'])\n"
            "### Feature-Driven Exploration Tools\n"
            "- 'expand_leaf_node_info(feature_path)'\n"
            "Given a feature path from the implemented feature tree, this tool expands and lists all\n"
            "associated interfaces (functions or classes) in a structural summary.\n"
            "*Usage*: Applied when analyzing how a specific functional leaf node in the design tree maps to\n"
            "repository interfaces.\n"
            "*Example*:\n"
            "expand_leaf_node_info('Algorithm/Supervised Learning/Classification Algorithms/Naive Bayes')\n"
            "- 'search_interface_by_functionality(keywords)'\n"
            "Performs a fuzzy semantic search for interfaces based on given keywords and returns the top-5\n"
            "most relevant interface implementations.\n"
            "*Usage*: Useful when the exact file or interface name is unknown, but functionality-related\n"
            "keywords are available.\n"
            "*Example*:\n"
            "search_interface_by_functionality(['optimize', 'initialize'])\n"
            "### Termination Tool\n"
            "- 'Terminate(result)'\n"
            "Terminates the localization exploration and returns the final ranked list of located interfaces.\n"
            "The result must follow the specified JSON-style format, including file name and interface\n"
            "type (function, class, or method).\n"
            "*Usage*: Invoked after completing exploration to deliver the final interface localization results.\n"
            "*Example*:\n"
            "Terminate(result=[\n"
            "{\"file\": \"top1_file_fullpath.py\", \"interface\": \"method: Class1.function1\"},\n"
            "{\"file\": \"top2_file_fullpath.py\", \"interface\": \"function: function2\"},\n"
            "{\"file\": \"top3_file_fullpath.py\", \"interface\": \"class: Class3\"},\n"
            "])\n"
            "\n"
            "Response Format (JSON only, no extra text):\n"
            "You must respond with a single JSON object and no leading/trailing text.\n"
            "If your previous response was invalid, re-output JSON only.\n"
            "{\n"
            "  \"tool_calls\": [\n"
            "    {\"tool\": \"view_file_interface_feature_map\", \"args\": {\"file_path\": \"path\"}},\n"
            "    {\"tool\": \"get_interface_content\", \"args\": {\"target_specs\": [\"path:QualName\"]}},\n"
            "    {\"tool\": \"expand_leaf_node_info\", \"args\": {\"feature_path\": \"A/B/C\"}},\n"
            "    {\"tool\": \"search_interface_by_functionality\", \"args\": {\"keywords\": [\"k1\", \"k2\"]}},\n"
            "    {\"tool\": \"Terminate\", \"args\": {\"result\": [{\"file\": \"...\", \"interface\": \"...\"}]}}\n"
            "  ]\n"
            "}\n"
        )

    def _parse_tool_calls(self, text: str) -> Optional[List[Dict[str, Any]]]:
        calls: List[Dict[str, Any]] = []
        stripped = text.strip()
        try:
            data = json.loads(stripped)
        except Exception:
            return None
        if isinstance(data, dict):
            tool_calls = data.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    name = call.get("tool") or call.get("name")
                    if not name:
                        continue
                    calls.append({"name": name, "args": call.get("args")})
                return calls
            if data.get("tool"):
                calls.append({"name": data.get("tool"), "args": data.get("args")})
                return calls
        return []

    def _execute_call(self, call: Dict[str, Any]) -> Any:
        name = call["name"]
        raw_args = call.get("raw_args", "")
        args = call.get("args")
        if name == "view_file_interface_feature_map":
            file_path = ""
            if isinstance(args, dict):
                file_path = args.get("file_path", "")
            if not file_path:
                file_path = self._parse_single_string(raw_args)
            return view_file_interface_feature_map(self._resolve_path(file_path))
        if name == "get_interface_content":
            if isinstance(args, dict):
                specs = args.get("target_specs", [])
            elif isinstance(args, list):
                specs = args
            else:
                specs = self._parse_literal(raw_args)
            normalized = self._normalize_specs(specs if isinstance(specs, list) else [])
            return get_interface_content(normalized)
        if name == "expand_leaf_node_info":
            if isinstance(args, dict):
                feature_path = args.get("feature_path", "")
            else:
                feature_path = self._parse_single_string(raw_args)
            return expand_leaf_node_info(feature_path, self.interfaces)
        if name == "search_interface_by_functionality":
            if isinstance(args, dict):
                keywords = args.get("keywords", [])
            elif isinstance(args, list):
                keywords = args
            else:
                keywords = self._parse_literal(raw_args)
            return search_interface_by_functionality(keywords if isinstance(keywords, list) else [], self.interfaces)
        if name == "Terminate":
            if isinstance(args, dict):
                parsed = args.get("result", [])
            elif isinstance(args, list):
                parsed = args
            else:
                parsed = self._parse_literal(raw_args.replace("result=", ""))
            normalized = []
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "file_path" in item and "file" not in item:
                        item["file"] = Path(item.pop("file_path")).name
                    elif isinstance(item, dict) and "file" in item:
                        item["file"] = Path(item["file"]).name
                    normalized.append(item)
            return Terminate(result=normalized if normalized else [])
        return None

    def _format_tool_result(self, call: Dict[str, Any], result: Any) -> str:
        return f"Tool result for {call['name']}:\n{result}"

    @staticmethod
    def _format_json_retry(response: str) -> str:
        return (
            "Your previous response was not valid JSON. "
            "Re-output a single JSON object with the required schema and no extra text."
        )

    @staticmethod
    def _log_llm_error(exc: Exception) -> None:
        message = str(exc).lower()
        context_markers = [
            "context length",
            "context window",
            "maximum context",
            "token limit",
            "too many tokens",
            "exceeds",
        ]
        if any(marker in message for marker in context_markers):
            logger.warning("LLM context/token limit issue: %s", exc)
        else:
            logger.error("LLM call failed: %s", exc)

    @staticmethod
    def _parse_literal(raw_args: str) -> Any:
        try:
            return ast.literal_eval(raw_args)
        except Exception:
            return raw_args

    @staticmethod
    def _parse_single_string(raw_args: str) -> str:
        arg = raw_args
        if "," in arg:
            arg = arg.split(",", 1)[0]
        return arg.strip().strip('"').strip("'")

    def _resolve_path(self, file_path: str) -> str:
        path = Path(file_path)
        if path.is_absolute():
            return str(path)
        return str(self.repo_root / file_path)

    def _normalize_specs(self, specs: List[str]) -> List[str]:
        normalized = []
        for spec in specs:
            if ":" not in spec:
                normalized.append(spec)
                continue
            file_path, qualname = spec.split(":", 1)
            normalized.append(f"{self._resolve_path(file_path)}:{qualname}")
        return normalized
