from __future__ import annotations

from pathlib import Path

import requests
import torch

from genome_lab.tokenizer import CharTokenizer


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def load_text(dataset: str, data_dir: str | Path) -> str:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if dataset != "tiny_shakespeare":
        raise ValueError(f"Unsupported dataset: {dataset}")

    target = data_dir / "tiny_shakespeare.txt"
    if not target.exists():
        resp = requests.get(TINY_SHAKESPEARE_URL, timeout=30)
        resp.raise_for_status()
        target.write_text(resp.text, encoding="utf-8")
    return target.read_text(encoding="utf-8")


def build_splits(
    text: str,
    train_split: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, CharTokenizer]:
    tokenizer = CharTokenizer.build(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx].to(device)
    val_data = data[split_idx:].to(device)
    return train_data, val_data, tokenizer


def get_batch(data: torch.Tensor, block_size: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, len(data) - block_size - 1, (batch_size,), device=data.device)
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
    return x, y

# === DEV LOG (real talk) ===
# - Problem: I kept hitting random issues like path errors, missing checkpoints, and runs that looked stuck.
# - Fix: Added safer defaults, auto best-run selection, stronger logging, and fallback config loading.
# - Another problem: training sometimes plateaus hard and feels like nothing is moving.
# - Fix: Keeping quick configs + evolution search so I can test changes faster instead of waiting forever.
# - What's going on now: this file is part of EvoLM's current prototype, and logs are here so future-me
#   (or anyone on GitHub) can see what broke, what got fixed, and why the code looks like this.
