#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from genome_lab.config import ExperimentConfig
from genome_lab.leaderboard import write_leaderboard
from genome_lab.logging_utils import log
from genome_lab.train import train_experiment


def pick_from_space(space: dict[str, Any]) -> dict[str, Any]:
    override: dict[str, Any] = {}
    for section, fields in space.items():
        section_override: dict[str, Any] = {}
        for key, values in fields.items():
            section_override[key] = random.choice(values)
        if section_override:
            override[section] = section_override
    return override


def get_nested(config: dict[str, Any], section: str, key: str) -> Any:
    return config.get(section, {}).get(key)


def set_nested(config: dict[str, Any], section: str, key: str, value: Any) -> None:
    config.setdefault(section, {})
    config[section][key] = value


def valid_model_fields(config: dict[str, Any]) -> bool:
    n_heads = int(get_nested(config, "model", "n_heads"))
    d_model = int(get_nested(config, "model", "d_model"))
    return d_model % n_heads == 0


def mutate_from_parent(
    parent: dict[str, Any],
    search_space: dict[str, Any],
    mutation_rate: float,
) -> dict[str, Any]:
    child: dict[str, Any] = {"model": dict(parent["model"]), "train": dict(parent["train"]), "data": dict(parent["data"])}
    for section, fields in search_space.items():
        for key, values in fields.items():
            if random.random() < mutation_rate:
                set_nested(child, section, key, random.choice(values))
    return child


def crossover(parent_a: dict[str, Any], parent_b: dict[str, Any], search_space: dict[str, Any]) -> dict[str, Any]:
    child: dict[str, Any] = {"model": {}, "train": {}, "data": {}}
    for section, fields in search_space.items():
        for key in fields.keys():
            source = parent_a if random.random() < 0.5 else parent_b
            set_nested(child, section, key, get_nested(source, section, key))
    return child


def make_child(
    parents: list[dict[str, Any]],
    search_space: dict[str, Any],
    mutation_rate: float,
    max_tries: int = 30,
) -> dict[str, Any]:
    for _ in range(max_tries):
        if len(parents) >= 2 and random.random() < 0.7:
            a, b = random.sample(parents, 2)
            candidate = crossover(a, b, search_space)
            candidate = mutate_from_parent(candidate, search_space, mutation_rate)
        else:
            p = random.choice(parents)
            candidate = mutate_from_parent(p, search_space, mutation_rate)

        if valid_model_fields(candidate):
            return candidate

    fallback = pick_from_space(search_space)
    if not valid_model_fields(fallback):
        raise RuntimeError("Could not generate a valid child config. Check d_model and n_heads search space.")
    return fallback


def flatten_runtime_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": dict(cfg["model"]),
        "train": dict(cfg["train"]),
        "data": dict(cfg["data"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-generation evolutionary search")
    parser.add_argument("--config", required=True, help="Path to evolution config JSON")
    args = parser.parse_args()
    evo_log = Path("outputs") / "evolve.log"

    evo_cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    base_cfg = ExperimentConfig.from_json(evo_cfg["base_config"]).raw
    log("evolve", f"Loaded evolution config from {args.config}", evo_log)
    log("evolve", f"Using base config {evo_cfg['base_config']}", evo_log)

    random.seed(evo_cfg.get("seed", 42))

    search_space = evo_cfg["search_space"]
    generations = int(evo_cfg["generations"])
    top_k = int(evo_cfg["top_k"])
    children_per_generation = int(evo_cfg["children_per_generation"])
    mutation_rate = float(evo_cfg["mutation_rate"])
    log(
        "evolve",
        f"Search settings generations={generations} top_k={top_k} children_per_generation={children_per_generation} mutation_rate={mutation_rate}",
        evo_log,
    )

    population_overrides = evo_cfg.get("population", [])
    if not population_overrides:
        population_overrides = [pick_from_space(search_space) for _ in range(children_per_generation)]

    all_results: list[dict[str, Any]] = []

    for generation in range(generations):
        log("evolve", f"=== Generation {generation} ===", evo_log)

        generation_results: list[dict[str, Any]] = []
        for idx, override in enumerate(population_overrides):
            run_cfg = ExperimentConfig(base_cfg).merged(override).raw
            run_cfg["run_name"] = f"gen{generation:02d}_cand{idx:02d}_{run_cfg['model']['n_layers']}L_{run_cfg['model']['n_heads']}H_{run_cfg['model']['d_model']}D"

            log("evolve", f"Running {run_cfg['run_name']}", evo_log)
            metrics = train_experiment(run_cfg)
            metrics["generation"] = generation
            metrics["genome"] = flatten_runtime_config(run_cfg)
            generation_results.append(metrics)
            all_results.append(metrics)

        generation_results.sort(key=lambda x: x["best_val_loss"])
        parents = generation_results[:top_k]

        log("evolve", "Top genomes this generation:", evo_log)
        for rank, row in enumerate(parents, start=1):
            log(
                "evolve",
                f"  #{rank} {row['run_name']} val_loss={row['best_val_loss']:.4f} "
                f"ppl={row['perplexity']:.2f} params={row['params']}",
                evo_log,
            )

        parent_genomes = [p["genome"] for p in parents]

        next_population: list[dict[str, Any]] = []
        while len(next_population) < children_per_generation:
            child = make_child(parent_genomes, search_space, mutation_rate)
            next_population.append(child)

        population_overrides = next_population

    leaderboard = write_leaderboard(all_results)

    log("evolve", "=== Global Leaderboard ===", evo_log)
    for idx, row in enumerate(leaderboard[:10], start=1):
        log(
            "evolve",
            f"#{idx} {row['run_name']} | val_loss={row['best_val_loss']:.4f} "
            f"eff={row['efficiency_score']:.4f} params={row['params']}",
            evo_log,
        )

    log("evolve", "Saved outputs/leaderboard.json and outputs/leaderboard.csv", evo_log)


if __name__ == "__main__":
    main()

# === DEV LOG (real talk) ===
# - Problem: I kept hitting random issues like path errors, missing checkpoints, and runs that looked stuck.
# - Fix: Added safer defaults, auto best-run selection, stronger logging, and fallback config loading.
# - Another problem: training sometimes plateaus hard and feels like nothing is moving.
# - Fix: Keeping quick configs + evolution search so I can test changes faster instead of waiting forever.
# - What's going on now: this file is part of EvoLM's current prototype, and logs are here so future-me
#   (or anyone on GitHub) can see what broke, what got fixed, and why the code looks like this.
