from __future__ import annotations

import sys
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


if __name__ == "__main__":
    uvicorn.run(
        "nexus_backend.main:app",
        host="127.0.0.1",
        port=7861,
        reload=False,
        log_level="warning",
        access_log=False,
    )
