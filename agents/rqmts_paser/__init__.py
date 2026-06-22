"""Legacy path ``agents.rqmts_paser`` → implementation in ``agents.ingest.rqmts_paser``."""

from __future__ import annotations

import agents.ingest.rqmts_paser as _impl

for _attr in list(vars(_impl)):
    if _attr.startswith("__"):
        continue
    globals()[_attr] = getattr(_impl, _attr)

del _impl, _attr
