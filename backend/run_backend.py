from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

from nexus_backend.security import require_explicit_network_ack


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


if __name__ == "__main__":
    host = os.environ.get("NEXUS_BACKEND_HOST", "127.0.0.1")
    try:
        require_explicit_network_ack(host)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    uvicorn.run(
        "nexus_backend.main:app",
        host=host,
        port=int(os.environ.get("NEXUS_BACKEND_PORT", "7861")),
        reload=False,
        log_level="warning",
        access_log=False,
    )
