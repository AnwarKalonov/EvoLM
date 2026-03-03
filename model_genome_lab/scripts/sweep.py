#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from genome_lab.config import ExperimentConfig
from genome_lab.leaderboard import write_leaderboard
from genome_lab.logging_utils import log
from genome_lab.train import train_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a genome sweep and produce leaderboard")
    parser.add_argument("--config", required=True, help="Path to sweep config JSON")
    args = parser.parse_args()
    sweep_log = Path("outputs") / "sweep.log"

    sweep_path = Path(args.config)
    sweep_cfg = json.loads(sweep_path.read_text(encoding="utf-8"))
    log("sweep", f"Loaded sweep config from {sweep_path}", sweep_log)

    base_cfg = ExperimentConfig.from_json(sweep_cfg["base_config"])
    log("sweep", f"Using base config {sweep_cfg['base_config']}", sweep_log)

    results = []
    for override in sweep_cfg["runs"]:
        run_cfg = base_cfg.merged(override).raw
        log("sweep", f"Running candidate {run_cfg['run_name']}", sweep_log)
        metrics = train_experiment(run_cfg)
        results.append(metrics)

    results = write_leaderboard(results)
    log("sweep", f"Sweep complete. Total runs={len(results)}", sweep_log)

    log("sweep", "Leaderboard:", sweep_log)
    for idx, row in enumerate(results, start=1):
        log(
            "sweep",
            f"#{idx} {row['run_name']} | val_loss={row['best_val_loss']:.4f} "
            f"ppl={row['perplexity']:.2f} params={row['params']} eff={row['efficiency_score']:.4f}",
            sweep_log,
        )

    log("sweep", "Saved outputs/leaderboard.json", sweep_log)
    log("sweep", "Saved outputs/leaderboard.csv", sweep_log)


if __name__ == "__main__":
    main()

# === DEV LOG (real talk) ===
# - Problem: I kept hitting random issues like path errors, missing checkpoints, and runs that looked stuck.
# - Fix: Added safer defaults, auto best-run selection, stronger logging, and fallback config loading.
# - Another problem: training sometimes plateaus hard and feels like nothing is moving.
# - Fix: Keeping quick configs + evolution search so I can test changes faster instead of waiting forever.
# - What's going on now: this file is part of EvoLM's current prototype, and logs are here so future-me
#   (or anyone on GitHub) can see what broke, what got fixed, and why the code looks like this.
