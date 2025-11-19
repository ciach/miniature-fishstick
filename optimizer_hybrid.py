#!/usr/bin/env python3
"""Hybrid GA + SA optimizer for Kaggle Santa layouts."""

import argparse
from pathlib import Path

from geometry import layout_cost
from ga_constructive_single import (
    build_layout_from_individual,
    run_ga_for_n,
    write_single_n_csv,
)
from optimizer_sa_parallel_imp import (
    HEAVY_SETTINGS,
    LIGHT_SETTINGS,
    SASettings,
    simulated_annealing_for_n,
)


def refine_with_sa(
    n: int,
    ga_population: int,
    ga_generations: int,
    sa_settings: SASettings,
):
    best, _ = run_ga_for_n(n, population_size=ga_population, generations=ga_generations)
    ga_trees = build_layout_from_individual(n, best)
    ga_cost = layout_cost(ga_trees)

    sa_trees, sa_cost = simulated_annealing_for_n(
        n, sa_settings, initial_layout=ga_trees
    )
    return ga_cost, sa_cost, ga_trees, sa_trees


def save_layout(n: int, trees, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_n{n}.csv"
    write_single_n_csv(n, trees, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid evolutionary + SA refinement for selected puzzle sizes."
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        required=True,
        help="Puzzle sizes to optimize (space separated).",
    )
    parser.add_argument("--ga-pop", type=int, default=20, help="GA population size.")
    parser.add_argument("--ga-gens", type=int, default=30, help="GA generations.")
    parser.add_argument(
        "--sa-mode",
        choices=["light", "heavy"],
        default="heavy",
        help="SA settings preset to use for refinement.",
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default=None,
        help="Optional directory to store refined layouts as CSVs.",
    )
    parser.add_argument(
        "--save-ga",
        action="store_true",
        help="Also save the GA-only layouts for inspection.",
    )

    args = parser.parse_args()

    sa_settings = HEAVY_SETTINGS if args.sa_mode == "heavy" else LIGHT_SETTINGS
    csv_dir = Path(args.csv_dir) if args.csv_dir else None

    for n in args.n:
        print(f"Running hybrid optimizer for n={n}...")
        ga_cost, sa_cost, ga_trees, sa_trees = refine_with_sa(
            n, args.ga_pop, args.ga_gens, sa_settings
        )
        improvement = ga_cost - sa_cost
        print(
            f"  GA seed cost: {ga_cost:.6f}\n"
            f"  SA refined cost: {sa_cost:.6f}\n"
            f"  Improvement: {improvement:.6f}"
        )

        if csv_dir:
            if args.save_ga:
                save_layout(n, ga_trees, csv_dir, "ga")
            save_layout(n, sa_trees, csv_dir, "hybrid")


if __name__ == "__main__":
    main()
