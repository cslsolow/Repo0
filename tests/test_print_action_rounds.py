import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "print_action_rounds.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("print_action_rounds", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_summary_includes_round_details_and_gap_add(tmp_path: Path):
    module = _load_script_module()

    (tmp_path / "actions_round_1.json").write_text(
        json.dumps(
            [
                {
                    "task": "ReqA",
                    "actions": [
                        {
                            "component": "Alpha",
                            "action": "split",
                            "rationale": "too broad",
                            "action_origin": "metric_split",
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "action_refinement_round_1.json").write_text(
        json.dumps({"round": 1, "stats": {"merge_group_count": 0, "split_group_count": 1, "action_counts": {"split": 1}}}),
        encoding="utf-8",
    )
    (tmp_path / "component_refinement_report_round_1.json").write_text(
        json.dumps(
            {
                "parents": [
                    {
                        "parent_task": "ReqA",
                        "split_report": {
                            "split_groups": [
                                {
                                    "component_name": "Alpha",
                                    "split_into": ["AlphaCore", "AlphaAPI"],
                                    "confidence": 0.9,
                                    "reason": "clear separation",
                                }
                            ]
                        },
                        "merge_report": {"merge_groups": []},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "gap_addition_report.json").write_text(
        json.dumps(
            {
                "accepted_count": 1,
                "parents": [
                    {
                        "parent_requirement": "ReqB",
                        "accepted": True,
                        "decision": "add_component",
                        "final_confidence": 0.81,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "actions.json").write_text(json.dumps([{"task": "ReqA", "actions": [{"component": "Alpha", "action": "split"}]}]), encoding="utf-8")

    summary = module.build_summary(tmp_path)

    assert "[proposed] actions_round_1.json" in summary
    assert "Alpha: split [metric_split]" in summary
    assert "[round] action_refinement_round_1.json" in summary
    assert "[accepted] component_refinement_report_round_1.json" in summary
    assert "Alpha -> ['AlphaCore', 'AlphaAPI']" in summary
    assert "[gap-add] gap_addition_report.json" in summary
    assert "ReqB: add_component" in summary
    assert "[final-actions]" in summary


def test_build_summary_falls_back_to_architecture_summary_when_no_round_files(tmp_path: Path):
    module = _load_script_module()

    (tmp_path / "architectures.json").write_text(
        json.dumps(
            [
                {
                    "parent_task": "ReqA",
                    "architecture": {
                        "sub_requirements": [{"name": "ReqA::core"}, {"name": "ReqA::api"}],
                        "components": [
                            {"name": "Alpha", "serves_subrequirements": ["ReqA::core"]},
                            {"name": "Beta", "serves_subrequirements": ["ReqA::api"]},
                        ],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    summary = module.build_summary(tmp_path)

    assert "[architecture-summary]" in summary
    assert "parents=1 components=2 subrequirements=2" in summary
    assert "No round artifacts found" not in summary
