import os
import argparse
import logging
from typing import Optional, List, Tuple
import json
import sys
import re
from agents.infra.llm_client import LLMClient
from agents.rqmts.graph import Graph

# --- Logging Setup ---
# Configure logging to output to stdout (which will be redirected to a log file by 'nohup > logfile.log')

# --- PROMPT (Requests Structured JSON Output) ---
PROMPT = """
You are an expert Software Architect. Your task is to analyze project requirements and construct a Directed Acyclic Graph (DAG) that captures **implementation-level dependencies** for optimal development ordering.

Core Dependency Analysis Framework (Implementation-Focused):

1. **Direct Code Dependency** - Does requirement B's implementation directly depend on A's code?
   - Does B need to import/call functions, classes, or modules from A?
   - Does B use data structures, types, or interfaces defined in A?
   - Does B extend/inherit from classes in A?
   - If yes, create edge: A → B

2. **Interface-Implementation Relationship** - Does B implement an abstraction defined in A?
   - Is A an abstract interface/protocol that B concretely implements?
   - Is A a generic framework that B specializes?
   - Is A a base class that B extends?
   - If yes, create edge: A → B (abstraction before implementation)

3. **Architectural Layering** - Does B build on top of A's layer?
   - Low-level primitives → High-level utilities
   - Core abstractions → Specific features
   - Backend systems → Frontend interfaces
   - Only create edge if B directly uses A's components (not just conceptual layering)
   - If yes, create edge: A → B

Critical Filtering Rules:
- **EXCLUDE** testing requirements (test suites, unit tests, integration tests, validation tests)
- **EXCLUDE** documentation requirements (specifications, guides, reference docs, tutorials, manuals)
- **EXCLUDE** auxiliary development tools (CI/CD, build scripts, linters) unless core to product
- **EXCLUDE** conceptual/semantic relationships that don't translate to code dependencies
- **EXCLUDE** execution flow dependencies unless they require direct API calls

Dependency Creation Best Practices (Strict Rules):
- Only create edge A → B if B's CODE will literally import/call/use A's code
- Ask: "Can B be implemented without A's code being written first?" If YES, don't create edge
- Identify the MOST DIRECT dependency - if B depends on C which depends on A, only create B → C
- Avoid transitive edges (if A → B and B → C exist, don't add A → C)
- Independent requirements that don't share code should have NO edges
- Use EXACT requirement names from input - no modifications
- When in doubt, prefer fewer edges (under-connecting is safer than over-connecting)

Edge Interpretation:
- A → B means "A must be implemented/understood before B can be implemented"
- A is the prerequisite/parent, B is the dependent/child
- Multiple edges can share the same parent or child

Output Requirements (strictly enforced):
1. Output ONLY a valid JSON object - no explanations, no markdown, no extra text
2. Format: {{"Parent Name": "Child Name", "Another Parent": "Another Child"}}
3. Use exact requirement names as they appear in input
4. One edge per key-value pair (if requirement has multiple children, use multiple pairs)
5. Valid JSON syntax - no trailing commas, properly escaped strings
6. If no valid edges exist after filtering, return empty object: {{}}
7. **CRITICAL**: Both parent and child in each edge MUST be exact requirement names from the input list below. Do NOT create edges with made-up or abbreviated names.

Input Project Requirements List:
{requirements_list}

Analyze the requirements and output the dependency edges JSON now.

Example JSON output format:
{{
    "Requirement A": "Requirement B",
    "Requirement C": "Requirement D"
}}
"""

INCREMENTAL_PROMPT = """
You are an expert Software Architect. Your task is to analyze project requirements and construct a Directed Acyclic Graph (DAG) that captures **implementation-level dependencies** for optimal development ordering.

Core Dependency Analysis Framework (Implementation-Focused):

1. **Direct Code Dependency** - Does requirement B's implementation directly depend on A's code?
   - Does B need to import/call functions, classes, or modules from A?
   - Does B use data structures, types, or interfaces defined in A?
   - Does B extend/inherit from classes in A?
   - If yes, create edge: A → B

2. **Interface-Implementation Relationship** - Does B implement an abstraction defined in A?
   - Is A an abstract interface/protocol that B concretely implements?
   - Is A a generic framework that B specializes?
   - Is A a base class that B extends?
   - If yes, create edge: A → B (abstraction before implementation)

3. **Architectural Layering** - Does B build on top of A's layer?
   - Low-level primitives → High-level utilities
   - Core abstractions → Specific features
   - Backend systems → Frontend interfaces
   - Only create edge if B directly uses A's components (not just conceptual layering)
   - If yes, create edge: A → B

Critical Filtering Rules:
- **EXCLUDE** testing requirements (test suites, unit tests, integration tests, validation tests)
- **EXCLUDE** documentation requirements (specifications, guides, reference docs, tutorials, manuals)
- **EXCLUDE** auxiliary development tools (CI/CD, build scripts, linters) unless core to product
- **EXCLUDE** conceptual/semantic relationships that don't translate to code dependencies
- **EXCLUDE** execution flow dependencies unless they require direct API calls

Dependency Creation Best Practices (Strict Rules):
- Only create edge A → B if B's CODE will literally import/call/use A's code
- Ask: "Can B be implemented without A's code being written first?" If YES, don't create edge
- Identify the MOST DIRECT dependency - if B depends on C which depends on A, only create B → C
- Avoid transitive edges (if A → B and B → C exist, don't add A → C)
- Independent requirements that don't share code should have NO edges
- Use EXACT requirement names from input - no modifications
- When in doubt, prefer fewer edges (under-connecting is safer than over-connecting)

Edge Interpretation:
- A → B means "A must be implemented/understood before B can be implemented"
- A is the prerequisite/parent, B is the dependent/child
- Multiple edges can share the same parent or child

Output Requirements (strictly enforced):
1. Output ONLY a valid JSON object - no explanations, no markdown, no extra text
2. Format: {{"Parent Name": "Child Name", "Another Parent": "Another Child"}}
3. Use exact requirement names as they appear in input
4. One edge per key-value pair (if requirement has multiple children, use multiple pairs)
5. Valid JSON syntax - no trailing commas, properly escaped strings
6. If no valid edges exist after filtering, return empty object: {{}}
7. **CRITICAL**: Both parent and child in each edge MUST be exact requirement names from the input list below. Do NOT create edges with made-up or abbreviated names.
8. **CRITICAL**: Each edge MUST include the NEW requirement as either parent or child.
9. Use the existing requirements and existing edges as context to connect the NEW requirement to the existing graph.

Input Project Requirements List:
{requirements_list}

Existing Edges (JSON):
{existing_edges}

New Requirement:
{new_requirement}

Analyze the requirements and output the dependency edges JSON now.

Example JSON output format:
{{
    "Requirement A": "Requirement B",
    "Requirement C": "Requirement D"
}}
"""

def read_file(file_path: str) -> Optional[str]:
    """Reads the content of a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            logging.info(f"Successfully read file: {file_path}")
            return f.read()
    except FileNotFoundError:
        logging.error(f"File not found at '{file_path}'. Script execution terminated.")
        return None


def _parse_edges_text(response_text: str) -> List[Tuple[str, str]]:
    """Parse JSON-like LLM response into (parent, child) pairs without json.loads."""
    response_text = response_text.strip()
    if not response_text:
        return []


    if "```json" in response_text:
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        if end != -1:
            response_text = response_text[start:end].strip()
    elif "```" in response_text:
        start = response_text.find("```") + 3
        end = response_text.find("```", start)
        if end != -1:
            response_text = response_text[start:end].strip()
    
    cleaned = response_text

    cleaned = cleaned.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    pairs: List[Tuple[str, str]] = []

    for match in re.finditer(r'"([^"]+)"\s*:\s*"([^"]+)"', cleaned):
        pairs.append((match.group(1).strip(), match.group(2).strip()))

    for match in re.finditer(r'"([^"]+)"\s*:\s*\[([^\]]*)\]', cleaned, re.DOTALL):
        parent = match.group(1).strip()
        children_raw = match.group(2)
        for child_match in re.finditer(r'"([^"]+)"', children_raw):
            pairs.append((parent, child_match.group(1).strip()))

    for match in re.finditer(
        r'"(?:parent|source)"\s*:\s*"([^"]+)"\s*,\s*"(?:child|target)"\s*:\s*"([^"]+)"',
        cleaned,
        re.DOTALL,
    ):
        pairs.append((match.group(1).strip(), match.group(2).strip()))

    return pairs

def generate_and_save_edges(requirements_content: str, output_dir: str, llm_client: LLMClient, output_filename: str = "edges.json"):
    """Calls the LLM API to generate dependency edges between requirements and saves them to the specified directory."""

    output_path = os.path.join(output_dir, output_filename)

    logging.info("Connecting to LLM API and generating dependency edges...")

    try:
        # 1. Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory ensured: {output_dir}")

        # Parse requirements to get valid requirement names
        requirements_data = json.loads(requirements_content)
        valid_requirement_names = set(
            req.get("name") for req in requirements_data.get("requirements", [])
            if req.get("name")
        )
        logging.info(f"Found {len(valid_requirement_names)} valid requirement names from input")

        # Combine Prompt
        full_prompt = PROMPT.replace("{requirements_list}", requirements_content)

        logging.info("Generating edges with LLM...")

        # 2. API Call using LLM client
        response_text = ""
        for attempt in range(2):
            try:
                response_text = llm_client.call(
                    messages=[
                        {"role": "system", "content": "You are an expert software architect who outputs strictly in JSON."},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=32768
                )
            except RuntimeError as e:
                logging.error(f"Failed to get edge-text response from LLM: {e}")
                return None
            if response_text and response_text.strip():
                break
            logging.warning(
                "Empty response from LLM while generating edges (attempt %d/2)",
                attempt + 1,
            )

        if not response_text or not response_text.strip():
            logging.error("Empty response from LLM after retry; no edges generated")
            return None

        # 3. Filter edges to ensure both parent and child are valid requirement names
        edge_pairs = _parse_edges_text(response_text)
        filtered_edges = {}
        invalid_edges = []
        
        for parent, child in edge_pairs:
            if parent not in valid_requirement_names:
                invalid_edges.append({"parent": parent, "child": child, "reason": "parent not in requirements list"})
                continue
            if child not in valid_requirement_names:
                invalid_edges.append({"parent": parent, "child": child, "reason": "child not in requirements list"})
                continue
            filtered_edges.setdefault(parent, []).append(child)
        
        # Log warnings for invalid edges
        if invalid_edges:
            logging.warning(f"Filtered out {len(invalid_edges)} invalid edges where nodes are not in the requirements list:")
            for invalid_edge in invalid_edges:
                logging.warning(f"  - Edge '{invalid_edge['parent']}' -> '{invalid_edge['child']}': {invalid_edge['reason']}")
        
        logging.info(
            "Valid edges after filtering: %d out of %d generated edges",
            sum(len(children) for children in filtered_edges.values()),
            len(edge_pairs),
        )

        # 4. Save the filtered edges to a file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_edges, f, ensure_ascii=False, indent=4)

        logging.info(f"Edge generation completed successfully. Results saved to: {output_path}")
        return filtered_edges

    except Exception as e:
        logging.error(f"An error occurred during the edge API call or file saving: {e}")
        return None


def generate_incremental_edges(
    requirements_content: str,
    existing_edges: dict,
    new_requirement: str,
    llm_client: LLMClient,
) -> Optional[dict]:
    """Generate dependency edges only involving the new requirement."""
    try:
        requirements_data = json.loads(requirements_content)
        valid_requirement_names = set(
            req.get("name") for req in requirements_data.get("requirements", [])
            if req.get("name")
        )
        valid_requirement_names.add(new_requirement.get("name"))
        if isinstance(new_requirement, dict):
            new_requirement_name = str(new_requirement.get("name", "")).strip()
        else:
            new_requirement_name = str(new_requirement).strip()
        if new_requirement_name not in valid_requirement_names:
            logging.error("New requirement '%s' not found in requirements list", new_requirement_name)
            return None

        prompt = INCREMENTAL_PROMPT
        prompt = prompt.replace("{requirements_list}", requirements_content)
        prompt = prompt.replace("{existing_edges}", json.dumps(existing_edges, ensure_ascii=False))
        if isinstance(new_requirement, dict):
            prompt = prompt.replace("{new_requirement}", json.dumps(new_requirement, ensure_ascii=False))
        else:
            prompt = prompt.replace("{new_requirement}", new_requirement_name)

        response_text = ""
        for attempt in range(2):
            try:
                response_text = llm_client.call(
                    messages=[
                        {"role": "system", "content": "You are an expert software architect who outputs strictly in JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
            except RuntimeError as e:
                logging.error(f"Failed to get text response from LLM: {e}")
                return None
            if response_text and response_text.strip():
                break
            logging.warning(
                "Empty response from LLM while generating incremental edges (attempt %d/2)",
                attempt + 1,
            )

        if not response_text or not response_text.strip():
            logging.error("Empty response from LLM after retry; no incremental edges generated")
            return None

        edge_pairs = _parse_edges_text(response_text)
        filtered_edges = {}
        invalid_edges = []
        for parent, child in edge_pairs:
            if not isinstance(parent, str) or not isinstance(child, str):
                invalid_edges.append({"parent": parent, "child": child, "reason": "parent/child not a string"})
                continue
            if parent not in valid_requirement_names:
                invalid_edges.append({"parent": parent, "child": child, "reason": "parent not in requirements list"})
                continue
            if child not in valid_requirement_names:
                invalid_edges.append({"parent": parent, "child": child, "reason": "child not in requirements list"})
                continue
            if new_requirement_name != parent and new_requirement_name != child:
                invalid_edges.append({"parent": parent, "child": child, "reason": "edge missing new requirement"})
                continue
            filtered_edges.setdefault(parent, []).append(child)

        if invalid_edges:
            logging.warning(
                "Filtered out %d invalid incremental edges:", len(invalid_edges)
            )
            for invalid_edge in invalid_edges:
                logging.warning(
                    "  - Edge '%s' -> '%s': %s",
                    invalid_edge["parent"],
                    invalid_edge["child"],
                    invalid_edge["reason"],
                )


        return filtered_edges
    except Exception as e:
        logging.error(f"Failed to generate incremental edges: {e}")
        return None


def build_graph_from_requirements_and_edges(
    requirements_data: dict,
    edges_data: dict,
) -> Graph:
    graph = Graph()

    for requirement in requirements_data.get("requirements", []):
        requirement_name = requirement.get("name")
        requirement_description = requirement.get("description", "")
        if not requirement_name:
            continue
        graph.add_node(requirement_name, requirement_description)

    for parent_name, child_value in edges_data.items():
        if parent_name not in graph.nodes:
            graph.add_node(parent_name)
        if isinstance(child_value, list):
            children = child_value
        else:
            children = [child_value]
        for child_name in children:
            if child_name not in graph.nodes:
                graph.add_node(child_name)
            graph.add_edge(parent_name, child_name)

    if graph.has_cycle():
        raise ValueError("The graph built from requirements.json and edges.json contains a cycle and is not a DAG.")

    return graph


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(
        description="Analyze requirements file using LLM to generate dependency edges for DAG construction."
    )

    # Positional argument for requirements file path
    parser.add_argument(
        "--req_path",
        type=str,
        help="Path to the requirements JSON file to be analyzed"
    )
    # Required keyword argument for output directory
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The required directory where the output edges.json file will be saved."
    )
    # Model name argument
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="",
        help="Optional reasoning_effort to pass through to the LLM API.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="The model name to use for the API call (e.g., gpt-4o, deepseek/DeepSeek-V3.2-Exp)."
    )
    # Base URL argument
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Optional base URL for the API endpoint (e.g., for custom proxies or local models like http://localhost:8000/v1)."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Optional API key for the service. If not provided, the script will use the API_KEY environment variable."
    )

    args = parser.parse_args()

    req_path = args.req_path
    output_dir = args.output_dir
    model_name = args.model
    reasoning_effort = args.reasoning_effort
    base_url = args.base_url
    api_key = args.api_key

    # Initialize LLM client
    llm_client = LLMClient({
        "api_key": api_key,
        "base_url": base_url,
        "model": model_name,
        "reasoning_effort": reasoning_effort
    }, output_dir, agent_name="graph_parser")

    requirements_content = read_file(req_path)

    if requirements_content:
        edges_content = generate_and_save_edges(requirements_content, output_dir, llm_client)
        if edges_content is not None:
            requirements_data = json.loads(requirements_content)
            graph = build_graph_from_requirements_and_edges(
                requirements_data,
                edges_content,
            )
            logging.info("Graph construction completed successfully")
            logging.info(graph.print_graph())
            logging.info(graph.to_dot())
    else:
        logging.info("Script execution terminated.")
