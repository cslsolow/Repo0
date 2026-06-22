"""Legacy path ``agents.pr_rqmts_paser`` → implementation in ``agents.ingest.pr_rqmts_paser``."""

from __future__ import annotations

import agents.ingest.pr_rqmts_paser as _impl

for _attr in list(vars(_impl)):
    if _attr.startswith("__"):
        continue
    globals()[_attr] = getattr(_impl, _attr)

del _impl, _attr
