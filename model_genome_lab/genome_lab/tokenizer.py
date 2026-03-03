from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def build(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos[t] for t in tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

# === DEV LOG (real talk) ===
# - Problem: I kept hitting random issues like path errors, missing checkpoints, and runs that looked stuck.
# - Fix: Added safer defaults, auto best-run selection, stronger logging, and fallback config loading.
# - Another problem: training sometimes plateaus hard and feels like nothing is moving.
# - Fix: Keeping quick configs + evolution search so I can test changes faster instead of waiting forever.
# - What's going on now: this file is part of EvoLM's current prototype, and logs are here so future-me
#   (or anyone on GitHub) can see what broke, what got fixed, and why the code looks like this.
