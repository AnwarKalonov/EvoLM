#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from genome_lab.config import ExperimentConfig
from genome_lab.logging_utils import log
from genome_lab.train import train_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Model Genome Lab run")
    parser.add_argument("--config", required=True, help="Path to run config JSON")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_json(args.config).raw
    run_log = Path("outputs") / "runs" / cfg["run_name"] / "run.log"
    log("train_one", f"Loaded config from {args.config}", run_log)
    metrics = train_experiment(cfg)

    log("train_one", "Final Metrics:", run_log)
    print(json.dumps(metrics, indent=2), flush=True)
    log("train_one", f"Saved at: {Path('outputs') / 'runs' / cfg['run_name']}", run_log)


if __name__ == "__main__":
    main()

# === DEV LOG (real talk) ===
# - Problem: I kept hitting random issues like path errors, missing checkpoints, and runs that looked stuck.
# - Fix: Added safer defaults, auto best-run selection, stronger logging, and fallback config loading.
# - Another problem: training sometimes plateaus hard and feels like nothing is moving.
# - Fix: Keeping quick configs + evolution search so I can test changes faster instead of waiting forever.
# - What's going on now: this file is part of EvoLM's current prototype, and logs are here so future-me
#   (or anyone on GitHub) can see what broke, what got fixed, and why the code looks like this.
