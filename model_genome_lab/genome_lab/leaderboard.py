from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def annotate_result(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    params = max(1, int(out["params"]))
    out["params_m"] = params / 1_000_000.0
    out["efficiency_score"] = out["best_val_loss"] + 0.02 * math.log10(params)
    return out


def write_leaderboard(results: list[dict[str, Any]], output_dir: str | Path = "outputs") -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    enriched = [annotate_result(r) for r in results]
    enriched.sort(key=lambda x: (x["best_val_loss"], x["params"]))

    leaderboard_json = output_dir / "leaderboard.json"
    leaderboard_csv = output_dir / "leaderboard.csv"

    leaderboard_json.write_text(json.dumps(enriched, indent=2), encoding="utf-8")

    with leaderboard_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "run_name",
                "best_val_loss",
                "perplexity",
                "params",
                "params_m",
                "efficiency_score",
                "elapsed_sec",
                "device",
            ],
        )
        writer.writeheader()
        for idx, row in enumerate(enriched, start=1):
            writer.writerow(
                {
                    "rank": idx,
                    "run_name": row["run_name"],
                    "best_val_loss": f"{row['best_val_loss']:.6f}",
                    "perplexity": f"{row['perplexity']:.4f}",
                    "params": row["params"],
                    "params_m": f"{row['params_m']:.3f}",
                    "efficiency_score": f"{row['efficiency_score']:.6f}",
                    "elapsed_sec": f"{row['elapsed_sec']:.2f}",
                    "device": row["device"],
                }
            )

    return enriched

# === DEV LOG (real talk) ===
# - Problem: I kept hitting random issues like path errors, missing checkpoints, and runs that looked stuck.
# - Fix: Added safer defaults, auto best-run selection, stronger logging, and fallback config loading.
# - Another problem: training sometimes plateaus hard and feels like nothing is moving.
# - Fix: Keeping quick configs + evolution search so I can test changes faster instead of waiting forever.
# - What's going on now: this file is part of EvoLM's current prototype, and logs are here so future-me
#   (or anyone on GitHub) can see what broke, what got fixed, and why the code looks like this.
