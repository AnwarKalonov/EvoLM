# EvoLM

EvoLM is an open-source, local-first training lab for building tiny language models from scratch and improving them through evolutionary search.

It is designed for students, indie builders, and early researchers who want to learn real model engineering internals, not just call APIs. EvoLM trains GPT-style decoder-only transformers from random initialization, compares architecture choices, and ranks results with reproducible metrics.

## Why EvoLM

Most beginner AI projects stop at wrappers. EvoLM focuses on core model work:

- Build and train your own transformer
- Explore architecture tradeoffs on consumer hardware
- Run automatic search across model "genes"
- Track quality vs size with a leaderboard
- Reproduce runs from config files

## Current Version (Prototype Scope)

This version includes:

- Character-level tokenizer pipeline
- Tiny Shakespeare dataset workflow
- From-scratch decoder-only transformer training
- Single-run training and checkpointing
- Multi-run sweep mode
- Multi-generation evolution mode (selection + mutation + crossover)
- Leaderboard generation (`csv` + `json`)
- Best-checkpoint text sampling
- Unified launcher CLI for beginner workflows

## How EvoLM Works

1. Define a base config (model, data, training).
2. Train one or many candidate models.
3. Evaluate each run on validation loss/perplexity.
4. Rank candidates by quality and efficiency.
5. Select top candidates as parents.
6. Create next generation via crossover + mutation.
7. Repeat and compare results over generations.

## Repository Layout

```text
configs/         Experiment configs (baseline, quick, sweep, evolution)
data/            Dataset files (downloaded locally)
genome_lab/      Core modules (model, data, training, leaderboard)
outputs/         Checkpoints, run configs, metrics, leaderboards
scripts/         Script entry points (train, sweep, evolve, sample)
run_lab.py       Unified CLI launcher (recommended for most users)
requirements.txt Python dependencies
```

## Requirements

- Python 3.9+
- PyTorch-compatible environment
- macOS, Linux, or Windows
- Optional GPU (Apple Silicon `mps` or NVIDIA `cuda`)

## Installation

```bash
cd "/path/to/EvoLM"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Quick Start (Recommended)

```bash
python3 run_lab.py demo
python3 run_lab.py status
python3 run_lab.py sample --best --prompt "ROMEO:"
```

What this does:

- `demo`: runs a quick evolutionary prototype config
- `status`: shows current run/leaderboard state
- `sample --best`: auto-selects best available checkpoint and generates text

## CLI Commands

```bash
# Fast prototype workflow
python3 run_lab.py demo

# Train one run
python3 run_lab.py train --config configs/baseline_quick.json

# Evolution search
python3 run_lab.py evolve --config configs/evolution_small.json

# Sample from best run
python3 run_lab.py sample --best --prompt "ROMEO:" --max-new-tokens 200

# Sample from a specific run
python3 run_lab.py sample --run-name baseline_30m --prompt "To be, or not to be:"

# Show status
python3 run_lab.py status
```

## Direct Script Usage (Advanced)

```bash
python3 scripts/train_one.py --config configs/baseline.json
python3 scripts/sweep.py --config configs/sweep_small.json
python3 scripts/evolve.py --config configs/evolution_small.json
python3 scripts/sample.py --best --prompt "ROMEO:"
```

## Config Files

Included configs:

- `configs/baseline.json`: fuller baseline run
- `configs/baseline_quick.json`: smaller/faster baseline
- `configs/sweep_small.json`: fixed multi-run sweep
- `configs/evolution_small.json`: deeper evolution search
- `configs/evolution_quick.json`: faster evolution prototype

Config sections:

- `run_name`: run identifier
- `seed`: reproducibility seed
- `device`: `auto`, `mps`, `cuda`, or `cpu`
- `data`: dataset/block/training split settings
- `model`: architecture genes (`n_layers`, `n_heads`, `d_model`, `dropout`)
- `train`: optimizer/training loop settings (`lr`, batch size, steps, eval cadence)
- `search_space` (evolution): allowed values for mutation/crossover

## Outputs

EvoLM writes to `outputs/`:

- `outputs/runs/<run_name>/best_model.pt`
- `outputs/runs/<run_name>/checkpoint_step_<N>.pt`
- `outputs/runs/<run_name>/metrics.json`
- `outputs/runs/<run_name>/run_config.json`
- `outputs/leaderboard.csv`
- `outputs/leaderboard.json`

## Metrics Explained

- `best_val_loss`: primary quality metric (lower is better)
- `perplexity`: language modeling quality proxy (lower is better)
- `params`: parameter count (size/cost)
- `params_m`: parameters in millions
- `efficiency_score`: quality-size tradeoff score

## Model and Data Notes

- Model type: decoder-only GPT-style transformer
- Attention: causal self-attention
- Tokenization: character-level (current prototype)
- Dataset: Tiny Shakespeare (auto-downloaded if missing)

## Apple Silicon Notes

- EvoLM auto-selects `mps` when available
- Start with quick configs first
- If memory issues occur, lower:
- `batch_size`
- `block_size`
- `d_model`
- `n_layers`

## Troubleshooting

Common issue: `zsh: command not found: python`  
Fix: use `python3` commands.

Common issue: checkpoint path errors  
Fix: use `python3 run_lab.py sample --best` to auto-pick checkpoint.

Common issue: interrupted run without final metrics  
Fix: EvoLM can still sample from checkpoint; use `--best` or `--run-name`.

Common warning: LibreSSL / urllib3 warning on macOS  
Fix: this warning is non-fatal for current workflow.

## Reproducibility

EvoLM improves reproducibility via:

- JSON config-driven runs
- saved `run_config.json` per run
- deterministic seed handling
- leaderboard artifacts for comparison

## Roadmap

- Subword tokenizer support (BPE/Unigram)
- More robust benchmark tasks
- Plotting tools for generation-by-generation progress
- Quantized export and faster inference paths
- Better experiment tracking/reporting UX

## Contribution Guidelines

Contributions are welcome. Good first contributions:

- New config presets
- New evaluation tasks
- Better data pipelines
- Visualization/reporting improvements
- Performance and memory optimizations

When opening a PR, include:

- problem statement
- change summary
- expected impact
- before/after metrics when applicable

## License

MIT License.

## Dev Log (Chill Notes)

- Problem: I kept running into checkpoint/path confusion while sampling.
- Fix: added auto best-run selection and safer fallback loading.
- Problem: runs would plateau and waste mad time.
- Fix: added quick configs + evolution mode so I can iterate way faster.
- Problem: hard to know what happened in long runs.
- Fix: added timestamped logs across training, evolve, sweep, sample, and launcher commands.
- Current vibe: EvoLM is stable enough to demo, still prototype-level, and now way easier to debug.
