"""Legacy path ``agents.graph_parser`` → implementation in ``agents.ingest.graph_parser``."""

from __future__ import annotations

import agents.ingest.graph_parser as _impl

for _attr in list(vars(_impl)):
    if _attr.startswith("__"):
        continue
    globals()[_attr] = getattr(_impl, _attr)

del _impl, _attr
