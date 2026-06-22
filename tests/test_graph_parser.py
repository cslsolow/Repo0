import pytest

from agents.graph_parser import _parse_edges_text


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"A": "B"}', {("A", "B")}),
        ('{"A": ["B", "C"]}', {("A", "B"), ("A", "C")}),
        (
            "```json\n{\n  \"A\": \"B\",\n  \"A\": \"C\"\n}\n```",
            {("A", "B"), ("A", "C")},
        ),
    ],
)
def test_parse_edges_text(text, expected):
    pairs = _parse_edges_text(text)
    print(pairs)
    assert set(pairs) == expected


def test_parse_edges_text_empty():
    assert _parse_edges_text("") == []
