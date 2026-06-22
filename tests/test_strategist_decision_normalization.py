import sys
from pathlib import Path

import pytest

pytest.importorskip("openai")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.strategist import StrategistAgent  # noqa: E402


def test_normalize_dag_decision_supports_delete_relation_type():
    agent = StrategistAgent(api_config={})
    decision = {
        "tag": "RELATION",
        "relation_type": "DELETE",
        "targets": ["OldReqA", "OldReqB"],
    }

    normalized = agent._normalize_dag_decision(decision)

    assert normalized["tag"] == "RELATION"
    assert normalized["relation_type"] == "DELETE"
    assert normalized["operation"] == "delete"
    assert normalized["affected_requirements"] == ["OldReqA", "OldReqB"]
