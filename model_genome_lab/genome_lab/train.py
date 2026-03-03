from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from genome_lab.data import build_splits, get_batch, load_text
from genome_lab.logging_utils import log
from genome_lab.model import GPTTiny


def pick_device(device_cfg: str) -> str:
    if device_cfg != "auto":
        return device_cfg
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_for_step(step: int, max_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(
    model: GPTTiny,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_steps: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    model.eval()
    for split_name, split_data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_steps, device=split_data.device)
        for k in range(eval_steps):
            x, y = get_batch(split_data, block_size, batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()
    model.train()
    return out


def train_experiment(config: dict) -> dict:
    set_seed(config["seed"])
    device = pick_device(config["device"])
    run_dir = Path("outputs") / "runs" / config["run_name"]
    run_log = run_dir / "run.log"
    log("train", f"Starting run '{config['run_name']}' on device={device}", run_log)

    text = load_text(config["data"]["dataset"], config["data"]["data_dir"])
    train_data, val_data, tokenizer = build_splits(
        text=text,
        train_split=config["data"]["train_split"],
        device=device,
    )
    log(
        "train",
        f"Loaded dataset={config['data']['dataset']} chars={len(text)} train_tokens={len(train_data)} val_tokens={len(val_data)}",
        run_log,
    )

    vocab_size = tokenizer.vocab_size
    config["model"]["vocab_size"] = vocab_size

    model = GPTTiny(
        vocab_size=vocab_size,
        block_size=config["data"]["block_size"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        d_model=config["model"]["d_model"],
        dropout=config["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
        betas=tuple(config["train"]["betas"]),
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    log("train", f"Saved run config to {run_dir / 'run_config.json'}", run_log)

    param_count = sum(p.numel() for p in model.parameters())
    log(
        "train",
        f"Model params={param_count} layers={config['model']['n_layers']} heads={config['model']['n_heads']} d_model={config['model']['d_model']}",
        run_log,
    )

    best_val = float("inf")
    t0 = time.time()

    for step in range(config["train"]["max_steps"]):
        lr = lr_for_step(
            step=step,
            max_steps=config["train"]["max_steps"],
            warmup_steps=config["train"]["warmup_steps"],
            base_lr=config["train"]["learning_rate"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        for _ in range(config["train"]["gradient_accumulation_steps"]):
            xb, yb = get_batch(
                train_data,
                block_size=config["data"]["block_size"],
                batch_size=config["train"]["batch_size"],
            )
            _, loss = model(xb, yb)
            (loss / config["train"]["gradient_accumulation_steps"]).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])
        optimizer.step()

        if step % config["train"]["eval_interval"] == 0 or step == config["train"]["max_steps"] - 1:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                block_size=config["data"]["block_size"],
                batch_size=config["train"]["batch_size"],
                eval_steps=config["train"]["eval_steps"],
            )
            train_loss, val_loss = losses["train"], losses["val"]
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), run_dir / "best_model.pt")
                log("train", f"New best checkpoint at step={step} val_loss={best_val:.4f}", run_log)

            log(
                "train",
                f"step={step:4d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={lr:.2e} best_val={best_val:.4f}",
                run_log,
            )

        if step % config["train"]["save_interval"] == 0 and step > 0:
            ckpt_path = run_dir / f"checkpoint_step_{step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            log("train", f"Saved checkpoint at step={step} -> {ckpt_path.name}", run_log)

    elapsed = time.time() - t0
    metrics = {
        "run_name": config["run_name"],
        "best_val_loss": best_val,
        "perplexity": math.exp(best_val),
        "elapsed_sec": elapsed,
        "device": device,
        "params": param_count,
        "model": config["model"],
        "train": config["train"],
        "data": config["data"],
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log(
        "train",
        f"Run finished: best_val_loss={metrics['best_val_loss']:.4f} perplexity={metrics['perplexity']:.2f} elapsed_sec={metrics['elapsed_sec']:.2f}",
        run_log,
    )
    log("train", f"Saved metrics to {run_dir / 'metrics.json'}", run_log)

    return metrics

# === DEV LOG (real talk) ===
# - Problem: I kept hitting random issues like path errors, missing checkpoints, and runs that looked stuck.
# - Fix: Added safer defaults, auto best-run selection, stronger logging, and fallback config loading.
# - Another problem: training sometimes plateaus hard and feels like nothing is moving.
# - Fix: Keeping quick configs + evolution search so I can test changes faster instead of waiting forever.
# - What's going on now: this file is part of EvoLM's current prototype, and logs are here so future-me
#   (or anyone on GitHub) can see what broke, what got fixed, and why the code looks like this.
