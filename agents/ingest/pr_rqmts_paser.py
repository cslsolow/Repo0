import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple
from agents.infra.llm_client import LLMClient

# --- Logging Setup ---
# Configure logging to output to stdout (which will be redirected to a log file by 'nohup > logfile.log')


PROMPT = """
You are a Senior Technical Product Manager and Code Auditor specializing in translating technical code changes (Git Diffs) into high-level, business-oriented requirement specifications.

**Task**
Analyze the provided Pull Request (PR) data (Title, Description, and Git Diff) to reverse-engineer the functional requirement that prompted these changes.

**Input Data**
{pr_content}

**ANALYSIS GUIDELINES**
1. **Capability-Level Extraction**: Focus on identifying the core technical capabilities or system components. Avoid delving into low-level code details, and emphasize the business functionality the system can now provide.
2. **Comprehensive Description**: For each identified capability, provide a thorough yet clear description that covers all relevant operations, decision branches, and business logic implemented in the diff.
3. **Excluding Noise**: Do not include any environmental setup, dependency installations, or configuration changes unless they affect core business logic or functionality.
4. **Context Awareness**: If 'PREVIOUSLY EXTRACTED REQUIREMENTS' are provided, ensure you focus only on new or modified capabilities, ignoring unchanged functionalities.

**Constraints:**
- The output must be a single, strictly valid JSON object.
- The output should avoid technical jargon, such as variable names, function signatures, or code-specific references. Focus on a business- and user-oriented language.
- The requirement should be captured at a capability level (e.g., "Automated Fraud Detection System"), rather than being overly specific.

**Output Schema:**
The JSON output should follow this structure:

{
  "requirement": "A high-level name describing the capability (e.g., 'Automated Fraud Detection System').",
  "requirement_type": "New Requirement" | "Modified Requirement" | "Non-functional Refactor",
  "summary": "A brief, clear summary outlining the capability’s purpose.",
  "details": "A detailed description of the capability, enumerating operations, business logic, rules, and any sub-features found in the diff. For example, 'The system evaluates fraud risk based on transaction velocity, historical user behavior, and geolocation anomalies. It performs real-time blocking of high-risk transactions and triggers alerts to the audit team.'",
  "affected_scope": "The functional modules or user journeys impacted by this capability (e.g., 'Transaction processing', 'User authentication').",
  "confidence_score": "A confidence score (0-10) reflecting how certain the analysis is of the identified capability."
}
"""

def read_file(file_path: str) -> Optional[str]:
    """Reads the content of any text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            logging.info(f"Successfully read file: {file_path}")
            return f.read()
    except FileNotFoundError:
        # Log error instead of printing
        logging.error(f"File not found at '{file_path}'. Script execution terminated.")
        return None
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return None

def parse_pr_items(raw_text: str) -> List[Dict[str, Any]]:
    """Parse PR items from JSON or JSONL input."""
    text = raw_text.strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if parsed is not None:
        logging.warning("Unsupported JSON content for PR input; falling back to JSONL parsing.")

    pr_items: List[Dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError:
            logging.warning("Skipping invalid JSONL line in PR input.")
            continue
        if isinstance(item, dict):
            pr_items.append(item)
        else:
            logging.warning("Skipping non-object JSONL line in PR input.")
    return pr_items


def get_pr_key(pr_item: Dict[str, Any]) -> Optional[str]:
    """Build a stable PR key for deduplication."""
    repo = pr_item.get("repository")
    pr_number = pr_item.get("pr_number")
    if repo and pr_number is not None:
        return f"{repo}#{pr_number}"
    owner = pr_item.get("owner")
    repo_name = pr_item.get("repo")
    if owner and repo_name and pr_number is not None:
        return f"{owner}/{repo_name}#{pr_number}"
    return None


def load_existing_requirements(jsonl_path: str) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Load existing requirements from JSONL and return records plus processed PR keys."""
    records: List[Dict[str, Any]] = []
    processed: Set[str] = set()
    if not jsonl_path or not os.path.exists(jsonl_path):
        return records, processed

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError:
                    logging.warning("Skipping invalid JSONL line in existing requirements.")
                    continue
                if isinstance(item, dict):
                    records.append(item)
                    pr_key = None
                    if len(item) == 1:
                        only_key = next(iter(item.keys()))
                        if isinstance(only_key, str) and "-" in only_key:
                            pr_key = only_key.replace("-", "#", 1)
                    if not pr_key:
                        pr_section = item.get("pr") if isinstance(item.get("pr"), dict) else item
                        pr_key = get_pr_key(pr_section)
                    if pr_key:
                        processed.add(pr_key)
    except Exception as e:
        logging.error("Error reading existing requirements file %s: %s", jsonl_path, e)

    return records, processed


def append_requirement_record(jsonl_path: str, record: Dict[str, Any]) -> None:
    """Append a single requirement record to JSONL output."""
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def generate_requirement_for_pr(pr_item: Dict[str, Any], llm_client: LLMClient) -> Optional[Dict[str, Any]]:
    """Call the LLM API to generate the requirement description for a PR."""
    pr_payload = json.dumps(pr_item, ensure_ascii=False, indent=2)
    full_prompt = PROMPT.replace("{pr_content}", pr_payload)
    logging.info(full_prompt)

    try:
        return llm_client.call_json(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert technical analyst specializing in software requirements engineering. Return only valid JSON matching the required schema."
                },
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.0
        )
    except RuntimeError as e:
        logging.error(f"Failed to get JSON response from LLM: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(
        description="Analyze PR data (JSONL input) using LLM to extract requirement descriptions."
    )

    # Positional argument for file path
    parser.add_argument(
        "--file_path",
        type=str,
        help="Path to the PR data file in JSONL format."
    )
    # Required keyword argument for output directory
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The required directory where output artifacts will be saved."
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Path to JSONL file where PR requirements will be appended."
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

    file_path = args.file_path
    output_dir = args.output_dir
    model_name = args.model
    reasoning_effort = args.reasoning_effort
    base_url = args.base_url
    api_key = args.api_key
    output_jsonl = args.output_jsonl or os.path.join(output_dir, "requirements.jsonl")

    # Initialize LLM client
    llm_client = LLMClient({
        "api_key": api_key,
        "base_url": base_url,
        "model": model_name,
        "reasoning_effort": reasoning_effort
    }, output_dir, agent_name="pr_rqmts_parser")

    # docs = extract_docstrings_from_directory(README_DIR)

    
    # input_readme = os.path.join(OUTPUT_DIR, 'docstring')
    # with open(input_readme, 'w', encoding='utf-8') as f:
    #     f.write(docs)

    file_content = read_file(file_path)

    if not file_content:
        logging.info("Script execution terminated.")
        raise SystemExit(1)

    pr_items = parse_pr_items(file_content)
    if not pr_items:
        logging.error("No valid PR entries found in input.")
        raise SystemExit(1)

    os.makedirs(output_dir, exist_ok=True)
    _, processed_keys = load_existing_requirements(output_jsonl)
    logging.info("Loaded %s processed PRs from %s", len(processed_keys), output_jsonl)

    appended = 0
    processed = 0
    for pr_item in pr_items:
        pr_key = get_pr_key(pr_item)
        if pr_key and pr_key in processed_keys:
            logging.info("Skipping already processed PR %s", pr_key)
            continue

        requirement = generate_requirement_for_pr(pr_item, llm_client)
        if requirement is None:
            continue

        record: Dict[str, Any]
        repo = pr_item.get("repository")
        pr_number = pr_item.get("pr_number")
        if repo and pr_number is not None:
            key = f"{repo}-{pr_number}"
            record = {key: requirement}
        else:
            record = {"pr": pr_item, "requirement": requirement}

        append_requirement_record(output_jsonl, record)
        appended += 1
        if pr_key:
            processed_keys.add(pr_key)
        processed += 1

    logging.info("Appended %s requirements to %s", appended, output_jsonl)
