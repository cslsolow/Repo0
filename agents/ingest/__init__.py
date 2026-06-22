"""README / PR requirement extraction and graph edge generation."""

from .graph_parser import generate_and_save_edges
from .rqmts_paser import generate_and_save_one_requirements

__all__ = ["generate_and_save_edges", "generate_and_save_one_requirements"]
