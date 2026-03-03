from __future__ import annotations

from datetime import datetime
from pathlib import Path


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_log(scope: str, message: str) -> str:
    return f"[{_stamp()}] [{scope}] {message}"


def log(scope: str, message: str, log_path: str | Path | None = None) -> None:
    line = format_log(scope, message)
    print(line, flush=True)
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

# === DEV LOG (real talk) ===
# - Problem: I kept hitting random issues like path errors, missing checkpoints, and runs that looked stuck.
# - Fix: Added safer defaults, auto best-run selection, stronger logging, and fallback config loading.
# - Another problem: training sometimes plateaus hard and feels like nothing is moving.
# - Fix: Keeping quick configs + evolution search so I can test changes faster instead of waiting forever.
# - What's going on now: this file is part of EvoLM's current prototype, and logs are here so future-me
#   (or anyone on GitHub) can see what broke, what got fixed, and why the code looks like this.
