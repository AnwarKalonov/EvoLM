#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from genome_lab.config import ExperimentConfig
from genome_lab.data import build_splits, load_text
from genome_lab.logging_utils import log
from genome_lab.model import GPTTiny


def load_best_run_name(outputs_dir: Path) -> str:
    leaderboard = outputs_dir / "leaderboard.json"
    if leaderboard.exists():
        rows = json.loads(leaderboard.read_text(encoding="utf-8"))
        if rows:
            return rows[0]["run_name"]

    runs_dir = outputs_dir / "runs"
    metrics_files = sorted(runs_dir.glob("*/metrics.json"))
    if not metrics_files:
        checkpoint_dirs = [p.parent for p in runs_dir.glob("*/best_model.pt")]
        if checkpoint_dirs:
            checkpoint_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return checkpoint_dirs[0].name
        raise FileNotFoundError("No leaderboard, metrics, or checkpoints found. Train a model first.")

    best_name = None
    best_loss = float("inf")
    for mf in metrics_files:
        row = json.loads(mf.read_text(encoding="utf-8"))
        if row["best_val_loss"] < best_loss:
            best_loss = row["best_val_loss"]
            best_name = row["run_name"]
    if best_name is None:
        raise RuntimeError("Could not find a best run.")
    return best_name


def build_cfg_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "device": metrics.get("device", "auto"),
        "data": metrics["data"],
        "model": metrics["model"],
    }


def load_run_cfg(run_dir: Path, fallback_config: str | None) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return build_cfg_from_metrics(json.loads(metrics_path.read_text(encoding="utf-8")))

    run_cfg_path = run_dir / "run_config.json"
    if run_cfg_path.exists():
        return json.loads(run_cfg_path.read_text(encoding="utf-8"))

    if fallback_config:
        return ExperimentConfig.from_json(fallback_config).raw

    baseline_cfg_path = Path("configs/baseline.json")
    if baseline_cfg_path.exists():
        if run_dir.name.startswith("baseline_"):
            return ExperimentConfig.from_json(baseline_cfg_path).raw

    match = re.search(r"_(\\d+)L_(\\d+)H_(\\d+)D$", run_dir.name)
    if match:
        if baseline_cfg_path.exists():
            cfg = ExperimentConfig.from_json(baseline_cfg_path).raw
            cfg["model"]["n_layers"] = int(match.group(1))
            cfg["model"]["n_heads"] = int(match.group(2))
            cfg["model"]["d_model"] = int(match.group(3))
            return cfg

    raise FileNotFoundError(
        f"No metrics/run_config for run '{run_dir.name}'. Pass --config for manual fallback."
    )


def pick_device(device_cfg: str) -> str:
    if device_cfg != "auto":
        return device_cfg
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample text from a trained model")
    parser.add_argument("--config", help="Path to run config JSON")
    parser.add_argument("--checkpoint", help="Path to checkpoint .pt")
    parser.add_argument("--run-name", help="Run folder name from outputs/runs")
    parser.add_argument("--best", action="store_true", help="Auto-select best run from leaderboard or metrics")
    parser.add_argument("--outputs-dir", default="outputs", help="Base outputs directory")
    parser.add_argument("--prompt", default="ROMEO:", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()
    sample_log = Path("outputs") / "sample.log"

    outputs_dir = Path(args.outputs_dir)
    cfg: dict[str, Any]
    checkpoint_path: Path
    selected_run_name = args.run_name

    if args.best:
        selected_run_name = load_best_run_name(outputs_dir)
        run_dir = outputs_dir / "runs" / selected_run_name
        cfg = load_run_cfg(run_dir, args.config)
        checkpoint_path = run_dir / "best_model.pt"
        log("sample", f"Auto-selected best run: {selected_run_name}", sample_log)
    elif args.run_name:
        run_dir = outputs_dir / "runs" / args.run_name
        cfg = load_run_cfg(run_dir, args.config)
        checkpoint_path = run_dir / "best_model.pt"
        log("sample", f"Using provided run: {args.run_name}", sample_log)
    else:
        if not args.config or not args.checkpoint:
            raise ValueError("Use either --best, --run-name, or provide both --config and --checkpoint.")
        cfg = ExperimentConfig.from_json(args.config).raw
        checkpoint_path = Path(args.checkpoint)
        log("sample", f"Using manual checkpoint: {checkpoint_path}", sample_log)

    device = pick_device(cfg["device"])
    log("sample", f"Sampling on device={device}", sample_log)

    text = load_text(cfg["data"]["dataset"], cfg["data"]["data_dir"])
    _, _, tokenizer = build_splits(text, cfg["data"]["train_split"], device)
    cfg["model"]["vocab_size"] = tokenizer.vocab_size

    model = GPTTiny(
        vocab_size=cfg["model"]["vocab_size"],
        block_size=cfg["data"]["block_size"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        d_model=cfg["model"]["d_model"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    log("sample", f"Loaded checkpoint {checkpoint_path}", sample_log)

    prompt_tokens = tokenizer.encode(args.prompt)
    x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        y = model.generate(x, max_new_tokens=args.max_new_tokens)

    output = tokenizer.decode(y[0].tolist())
    log("sample", f"Generated {args.max_new_tokens} tokens for prompt='{args.prompt}'", sample_log)
    print(
        json.dumps(
            {
                "prompt": args.prompt,
                "run_name": selected_run_name if selected_run_name else "manual",
                "checkpoint": str(checkpoint_path),
                "sample": output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

# === DEV LOG (real talk) ===
# - Problem: I kept hitting random issues like path errors, missing checkpoints, and runs that looked stuck.
# - Fix: Added safer defaults, auto best-run selection, stronger logging, and fallback config loading.
# - Another problem: training sometimes plateaus hard and feels like nothing is moving.
# - Fix: Keeping quick configs + evolution search so I can test changes faster instead of waiting forever.
# - What's going on now: this file is part of EvoLM's current prototype, and logs are here so future-me
#   (or anyone on GitHub) can see what broke, what got fixed, and why the code looks like this.
