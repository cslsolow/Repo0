import os
import argparse
import logging
from typing import Optional
import ast
from typing import List
import json
from agents.infra.llm_client import LLMClient

# --- Logging Setup ---
# Configure logging to output to stdout (which will be redirected to a log file by 'nohup > logfile.log')


PROMPT = """
Your are a Senior Technical Product Manager and Business Analyst. Your task is to analyze the following project documentation and extract all functional requirements.

CLASSIFICATION:
Requirements: The system must include a set of functionalities that are essential for performing its core tasks, while also optimizing performance, enhancing user experience, and improving overall efficiency. These functionalities work together to ensure the system operates effectively, reliably, and meets its intended objectives.

INSTRUCTIONS:
1. Infer requirements from project documentation, dependencies, features, and implementation details. EXCLUDE all environmental, setup, and general configuration settings.
2. Strictly output a single, valid JSON object that adheres precisely to the REQUIRED JSON SCHEMA provided below. 
4. Based on the 'PREVIOUSLY EXTRACTED REQUIREMENTS,' attempt to identify additional requirements that were not captured in previous rounds.

HIERARCHICAL REQUIREMENT STRUCTURE:
- Extract requirements at an appropriate level: MAJOR CAPABILITIES should be separate requirements
- For each major capability, if it has multiple related operations or features, include them comprehensively in the description field
- DO NOT split low-level operations into separate requirements 
- The description should capture all implementation details and sub-features of that capability

Each requirement should be:
1. **Capability-level**: Each requirement represents a distinct technical capability or system component (not too high, not too low)
2. **Complete**: The description must capture ALL operations, features, and details related to that capability
3. **Distinct**: Core technical capabilities should be separate requirements
4. **Comprehensive**: Within each requirement's description, enumerate all related operations and features
5. **Informative**: Preserve all relevant technical details from the documentation


[Project Documentation]: 
{documentation_content}

PREVIOUSLY EXTRACTED REQUIREMENTS:
{requirements_list}

REQUIRED JSON SCHEMA:
{
  "project_summary": "A concise summary of the project's purpose.",
  "requirements": [
    {
      "name": "string (Distinct technical capability or system component name)",
      "description": "string (Comprehensive description capturing all operations, features, and implementation details of this capability)",
    }
  ]
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

def extract_docstrings_from_directory(directory: str) -> str:
    """Recursively find and extract docstrings from all Python files in a directory and return them as a single string."""
    try:
        logging.info(f"Starting recursive analysis in directory: {directory}")

        all_docstrings = []  # List to hold all docstrings

        # Iterate over the directory and its subdirectories
        for root, _, files in os.walk(directory):
            for filename in files:
                # Check if the file is a Python file
                if filename.endswith('.py'):
                    file_path = os.path.join(root, filename)
                    # logging.info(f"Processing Python file: {file_path}")

                    # Read the content of the Python file and parse its AST
                    with open(file_path, 'r', encoding='utf-8') as file:
                        file_content = file.read()
                        tree = ast.parse(file_content)

                        # Iterate through all nodes in the AST and extract docstrings
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                                docstring = ast.get_docstring(node)
                                if docstring:
                                    all_docstrings.append(docstring)

        # Combine all docstrings into a single string, separating them by two newlines
        return "\n\n".join(all_docstrings) if all_docstrings else "No docstrings found."
    
    except Exception as e:
        logging.critical(f"A critical error occurred: {e}")
        return "Error occurred while extracting docstrings."



def find_and_read_files(file_dir: str) -> Optional[str]:
    """Recursively reads the content of all relevant text files in a directory and its subdirectories."""
    try:
        logging.info(f"Starting recursive analysis in directory: {file_dir}")
        
        file_contents = []  # List to hold content of all found files

        # Recursively walk through the directory and subdirectories
        for root, _, files in os.walk(file_dir):
            for filename in files:
                filename_lower = filename.lower()

                # Check if the file name contains "readme" and has a valid extension
                is_valid_file = "readme" in filename_lower or filename_lower.endswith((".md", ".txt", ".rst", ".req"))
                # is_valid_extension = filename_lower.endswith(".rst") or filename_lower.endswith(".md")
                # is_valid_extension = filename_lower.endswith(".req")

                if is_valid_file:
                    file_full_path = os.path.join(root, filename)

                    logging.info(f"\n--- Processing file: {filename} at {root} ---")

                    file_content = read_file(file_full_path)

                    if file_content:
                        file_contents.append(file_content)

        if file_contents:
            return "\n\n".join(file_contents)  # Join the content of all files
        else:
            logging.warning(f"No relevant files found in directory: {file_dir}")
            return None
    except FileNotFoundError:
        logging.error(f"Input directory not found: {file_dir}. Script terminated.")
        return None
    except Exception as e:
        logging.critical(f"A critical error occurred in the main loop: {e}")
        return None
    
def generate_and_save_one_requirements(readme_content: str, output_dir: str, llm_client: LLMClient, output_filename: str = "requirements.json", max_iterations: int = 5):
    """Calls the LLM API to generate the requirements, supports multi-turn processing, and saves them to the specified directory."""

    output_path = os.path.join(output_dir, output_filename)

    logging.info("Connecting to LLM API and generating requirements...")

    def _normalize_requirement_item(item, index: int) -> Optional[dict]:
        if not isinstance(item, dict):
            logging.warning("Skipping malformed requirement item at index %s: expected dict, got %s", index, type(item).__name__)
            return None
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        if not name:
            fallback_name = str(item.get("title", "")).strip()
            if fallback_name:
                name = fallback_name
                logging.warning("Requirement item at index %s missing 'name'; using 'title' as fallback", index)
        if not name:
            logging.warning("Skipping requirement item at index %s because 'name' is missing", index)
            return None
        return {
            "name": name,
            "description": description,
        }

    try:
        # 1. Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory ensured: {output_dir}")

        # Initialize the list of extracted requirements (empty at the start)
        all_requirements = []

        requirements_data = None

        # Combine Prompt
        full_prompt = PROMPT.replace("{documentation_content}", readme_content)

        for iteration in range(max_iterations):
            logging.info(f"Iteration {iteration + 1}...")

            # If there are existing requirements, append them to the prompt
            requirements_list = json.dumps(all_requirements, ensure_ascii=False, indent=4)

            full_prompt_cp = full_prompt
            # Update the full prompt with previously extracted requirements
            full_prompt_cp = full_prompt_cp.replace("{requirements_list}", requirements_list)

            if iteration == 0:
                logging.info(full_prompt_cp)
            else:
                logging.info(requirements_list)


            # 2. API Call using LLM client
            try:
                requirements_data = llm_client.call_json(
                    messages=[
                        {"role": "system", "content": "You are an expert technical analyst specializing in software requirements engineering. Extract functional and non-functional requirements from technical documentation. Focus on concrete, implementable requirements that can be verified through testing."},
                        {"role": "user", "content": full_prompt_cp}
                    ],
                    temperature=0.0,
                    max_tokens=32768
                )
            except RuntimeError as e:
                logging.error(f"Failed to get JSON response from LLM: {e}")
                break

            # Handle case where LLM returns a list instead of dict
            if isinstance(requirements_data, list):
                logging.warning(requirements_data)
                logging.warning("LLM returned list instead of dict for requirements, treating as requirements list")
                raw_requirements = requirements_data
            else:
                # Update the primary and secondary requirements
                raw_requirements = requirements_data.get("requirements", [])

            new_primary_requirements = []
            for idx, req in enumerate(raw_requirements):
                normalized_req = _normalize_requirement_item(req, idx)
                if normalized_req is not None:
                    new_primary_requirements.append(normalized_req)

            # Avoid adding duplicate primary requirements by comparing the 'key' (name) in the dictionary
            deduped_requirements = []
            seen_names = {str(existing_req.get("name", "")).strip() for existing_req in all_requirements if isinstance(existing_req, dict)}
            for req in new_primary_requirements:
                req_name = req["name"]
                if req_name in seen_names:
                    continue
                deduped_requirements.append(req)
                seen_names.add(req_name)
            new_primary_requirements = deduped_requirements


            if not new_primary_requirements or len(new_primary_requirements) == 0:
                logging.info("No new requirements found in this iteration. Stopping.")
                break

            # Add the newly extracted requirements to the ongoing list
            all_requirements.extend(new_primary_requirements)

            logging.info(f"Found {len(new_primary_requirements)} new primary requirements.")


        if requirements_data is None:
            raise RuntimeError("Requirement extraction failed: no response data was produced by the parser")
        if not isinstance(requirements_data, dict):
            requirements_data = {}
        requirements_data["requirements"] = all_requirements

        # 3. Save the final requirements to a file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(requirements_data, f, ensure_ascii=False, indent=4)

        input_readme = os.path.join(output_dir, 'README')
        with open(input_readme, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        logging.info(f"Requirement extraction completed successfully. Results saved to: {output_path}")

    except Exception as e:
        logging.error(f"An error occurred during the API call or file saving: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(
        description="Analyze any text file using LLM to extract and classify requirements, outputting in JSON format to a specified directory."
    )

    # Positional argument for file path
    parser.add_argument(
        "--file_path",
        type=str,
        help="Path to the text file to be analyzed (e.g., ./docs/README.md, ./requirements.txt)"
    )
    # Required keyword argument for output directory
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The required directory where the output file (requirements.json) will be saved."
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

    # Initialize LLM client
    llm_client = LLMClient({
        "api_key": api_key,
        "base_url": base_url,
        "model": model_name,
        "reasoning_effort": reasoning_effort
    }, output_dir, agent_name="rqmts_parser")

    # docs = extract_docstrings_from_directory(README_DIR)

    
    # input_readme = os.path.join(OUTPUT_DIR, 'docstring')
    # with open(input_readme, 'w', encoding='utf-8') as f:
    #     f.write(docs)

    file_content = read_file(file_path)

    if file_content:
        # Pass output_dir to the generator function
        generate_and_save_one_requirements(file_content, output_dir, llm_client)
    else:
        # Logging script termination after file read failure
        logging.info("Script execution terminated.")
