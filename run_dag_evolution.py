"""Integration example: Using DAG Evolution in the main pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents import (
    DAGEvolutionAgent,
    MemoryAgent,
    RequirementDAG,
    RequirementNode,
    StrategistAgent,
)
from run_agents import build_evolution_action_override


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incremental DAG construction with evolution"
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Repository name"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).parent,
        help="Workspace root"
    )
    parser.add_argument(
        "--new-requirements",
        type=Path,
        required=True,
        help="JSON file with new requirements to add"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key for LLM"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", ""),
        help="API base URL"
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="",
        help="Optional reasoning_effort to pass through to the LLM API.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DeepSeek-V3.2-Exp",
        help="Model name"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    workspace = args.workspace.resolve()
    repo_root = workspace / "repos" / args.repo
    
    if not repo_root.exists():
        raise FileNotFoundError(f"Repository not found: {repo_root}")
    
    # Load existing DAG
    print(f"Loading existing DAG for {args.repo}...")
    dag = RequirementDAG.from_repo(repo_root)
    print(f"Initial DAG: {dag.summary()}\n")
    
    # Setup agents
    api_config = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
    }
    
    memory_agent = MemoryAgent(workspace)
    memory_agent.build_memory(args.repo)
    
    evolution_agent = DAGEvolutionAgent(dag, memory_agent, api_config)
    strategist = StrategistAgent(api_config)
    
    # Load new requirements to add
    if not args.new_requirements.exists():
        raise FileNotFoundError(f"Requirements file not found: {args.new_requirements}")
    
    with open(args.new_requirements, 'r', encoding='utf-8') as f:
        new_reqs_data = json.load(f)
    
    if not isinstance(new_reqs_data, list):
        new_reqs_data = [new_reqs_data]
    
    # Process each new requirement
    print(f"\nProcessing {len(new_reqs_data)} new requirements...\n")
    
    for i, req_data in enumerate(new_reqs_data, 1):
        print(f"--- Requirement {i}/{len(new_reqs_data)} ---")
        print(f"Name: {req_data.get('name', 'Unknown')}")
        print(f"Description: {req_data.get('description', '')[:100]}...")
        
        # Create requirement node
        new_req = RequirementNode(
            name=req_data.get("name", f"Requirement_{i}"),
            description=req_data.get("description", ""),
            metadata=req_data.get("metadata", {})
        )
        
        # Let strategist decide the operation
        decision = strategist.choose_dag_operation(
            req_data,
            dag,
        )
        action_override = build_evolution_action_override(decision)
        if action_override is None:
            print("Decision: EXISTING")
            print("Reason: Requirement already covered, skipping")
            print()
            continue
        
        print(f"Decision: {decision.get('operation', 'add').upper()}")
        print(f"Reason: {decision.get('reason', 'N/A')}")
        
        # Execute the operation via evolution agent
        result = evolution_agent.add_sub_requirement(
            new_req,
            parent_names=action_override.get("suggested_parents", []),
            context=decision.get("reason", ""),
            action_override=action_override,
        )
        
        if result.get("success"):
            print(f"✓ Success: {result['action']}")
            print(f"  New nodes: {', '.join(result['new_nodes'])}")
            print(f"  Affected: {', '.join(result['affected_nodes'])}")
            
            # Record in memory
            if result.get("operation_record"):
                memory_agent.record_dag_operation(result["operation_record"])
        else:
            print(f"✗ Failed: {result.get('error', 'Unknown error')}")
        
        print()
    
    # Save updated DAG
    output_dir = repo_root / "agents_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export DAG
    with open(output_dir / "evolved_dag.json", 'w', encoding='utf-8') as f:
        f.write(json.dumps(dag.to_dict(), indent=2))
    
    # Export evolution state
    state = evolution_agent.export_state()
    with open(output_dir / "evolution_state.json", 'w', encoding='utf-8') as f:
        f.write(json.dumps(state, indent=2))
    
    # Save memory
    memory_agent.persist(output_dir)
    
    print(f"\n=== Final Results ===")
    print(f"Final DAG: {dag.summary()}")
    print(f"Total operations: {len(evolution_agent.operation_history)}")
    print(f"\nOutputs saved to: {output_dir}")
    
    print("\nOperation History:")
    for i, op in enumerate(evolution_agent.get_operation_history(), 1):
        print(f"{i}. {op['operation_type'].upper()}: {op['reason']}")


if __name__ == "__main__":
    main()
