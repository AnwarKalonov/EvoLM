#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from genome_lab.logging_utils import log

ROOT = Path(__file__).resolve().parent


def run_cmd(args: list[str]) -> None:
    log("run_lab", f"$ {' '.join(args)}", ROOT / "outputs" / "run_lab.log")
    subprocess.run(args, check=True, cwd=ROOT)


def cmd_train(args: argparse.Namespace) -> None:
    run_cmd([sys.executable, "scripts/train_one.py", "--config", args.config])


def cmd_evolve(args: argparse.Namespace) -> None:
    run_cmd([sys.executable, "scripts/evolve.py", "--config", args.config])


def cmd_demo(_: argparse.Namespace) -> None:
    run_cmd([sys.executable, "scripts/evolve.py", "--config", "configs/evolution_quick.json"])


def cmd_sample(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "scripts/sample.py",
        "--prompt",
        args.prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.best:
        cmd.append("--best")
    elif args.run_name:
        cmd.extend(["--run-name", args.run_name])
    else:
        cmd.extend(["--best"])
    cmd.extend(["--config", "configs/baseline.json"])
    run_cmd(cmd)


def cmd_status(_: argparse.Namespace) -> None:
    leaderboard = ROOT / "outputs" / "leaderboard.csv"
    runs_dir = ROOT / "outputs" / "runs"

    if leaderboard.exists():
        with leaderboard.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        log("run_lab", f"Leaderboard: {leaderboard}", ROOT / "outputs" / "run_lab.log")
        log("run_lab", f"Total ranked runs: {len(rows)}", ROOT / "outputs" / "run_lab.log")
        if rows:
            top = rows[0]
            log("run_lab", "Best run:", ROOT / "outputs" / "run_lab.log")
            log("run_lab", f"  run_name: {top['run_name']}", ROOT / "outputs" / "run_lab.log")
            log("run_lab", f"  val_loss: {top['best_val_loss']}", ROOT / "outputs" / "run_lab.log")
            log("run_lab", f"  perplexity: {top['perplexity']}", ROOT / "outputs" / "run_lab.log")
            log("run_lab", f"  params: {top['params']}", ROOT / "outputs" / "run_lab.log")
            log("run_lab", f"  efficiency_score: {top.get('efficiency_score', 'n/a')}", ROOT / "outputs" / "run_lab.log")
        return

    if runs_dir.exists():
        metrics_files = sorted(runs_dir.glob("*/metrics.json"))
        if metrics_files:
            log("run_lab", "No leaderboard yet. Found runs:", ROOT / "outputs" / "run_lab.log")
            for m in metrics_files:
                row = json.loads(m.read_text(encoding="utf-8"))
                log(
                    "run_lab",
                    f"  {row['run_name']} val_loss={row['best_val_loss']:.4f} ppl={row['perplexity']:.2f}",
                    ROOT / "outputs" / "run_lab.log",
                )
            return
        checkpoint_files = sorted(runs_dir.glob("*/best_model.pt"))
        if checkpoint_files:
            log(
                "run_lab",
                "Found checkpoints (runs may have been interrupted before final metrics):",
                ROOT / "outputs" / "run_lab.log",
            )
            for ckpt in checkpoint_files:
                log("run_lab", f"  {ckpt.parent.name}", ROOT / "outputs" / "run_lab.log")
            log("run_lab", "Try: python3 run_lab.py sample --best --prompt \"ROMEO:\"", ROOT / "outputs" / "run_lab.log")
            return

    log("run_lab", "No runs found yet. Start with: python3 run_lab.py demo", ROOT / "outputs" / "run_lab.log")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EvoLM launcher")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run a quick evolutionary prototype")
    p_demo.set_defaults(func=cmd_demo)

    p_train = sub.add_parser("train", help="Train one run")
    p_train.add_argument("--config", default="configs/baseline_quick.json")
    p_train.set_defaults(func=cmd_train)

    p_evolve = sub.add_parser("evolve", help="Run evolution search")
    p_evolve.add_argument("--config", default="configs/evolution_small.json")
    p_evolve.set_defaults(func=cmd_evolve)

    p_sample = sub.add_parser("sample", help="Sample text from best or specific run")
    p_sample.add_argument("--best", action="store_true")
    p_sample.add_argument("--run-name")
    p_sample.add_argument("--prompt", default="ROMEO:")
    p_sample.add_argument("--max-new-tokens", type=int, default=200)
    p_sample.set_defaults(func=cmd_sample)

    p_status = sub.add_parser("status", help="Show current best run and metrics")
    p_status.set_defaults(func=cmd_status)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

# === DEV LOG (real talk) ===
# - Problem: I kept hitting random issues like path errors, missing checkpoints, and runs that looked stuck.
# - Fix: Added safer defaults, auto best-run selection, stronger logging, and fallback config loading.
# - Another problem: training sometimes plateaus hard and feels like nothing is moving.
# - Fix: Keeping quick configs + evolution search so I can test changes faster instead of waiting forever.
# - What's going on now: this file is part of EvoLM's current prototype, and logs are here so future-me
#   (or anyone on GitHub) can see what broke, what got fixed, and why the code looks like this.
