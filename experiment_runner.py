#!/usr/bin/env python3
"""
Experiment runner for tuning GA + SA hyperparameters.
"""

import argparse
import time
import pandas as pd
from rich.console import Console
from rich.table import Table

from optimizer_sa_parallel_imp import HEAVY_SETTINGS, LIGHT_SETTINGS
from optimizer_hybrid import refine_with_sa

console = Console()

def run_experiment(ns: list[int]):
    # Hyperparameter grid
    ga_pops = [20, 50, 100]
    ga_gens = [20, 50]
    sa_modes = ["light", "heavy"]

    results = []

    total_runs = len(ns) * len(ga_pops) * len(ga_gens) * len(sa_modes)
    current_run = 0

    console.print(f"[bold cyan]Starting experiments: {total_runs} total runs[/bold cyan]")

    for n in ns:
        for pop in ga_pops:
            for gens in ga_gens:
                for mode in sa_modes:
                    current_run += 1
                    sa_settings = HEAVY_SETTINGS if mode == "heavy" else LIGHT_SETTINGS
                    
                    console.print(f"[{current_run}/{total_runs}] Running n={n}, pop={pop}, gens={gens}, sa={mode}...")
                    
                    start_time = time.time()
                    try:
                        ga_cost, sa_cost, _, _ = refine_with_sa(
                            n, pop, gens, sa_settings
                        )
                        elapsed = time.time() - start_time
                        
                        results.append({
                            "n": n,
                            "ga_pop": pop,
                            "ga_gens": gens,
                            "sa_mode": mode,
                            "runtime_sec": round(elapsed, 2),
                            "ga_cost": round(ga_cost, 6),
                            "final_cost": round(sa_cost, 6),
                            "improvement": round(ga_cost - sa_cost, 6)
                        })
                    except Exception as e:
                        console.print(f"[red]Run failed: {e}[/red]")

    # Create summary table
    table = Table(title="Experiment Results")
    table.add_column("n", justify="right", style="cyan")
    table.add_column("GA Pop", justify="right")
    table.add_column("GA Gens", justify="right")
    table.add_column("SA Mode", style="magenta")
    table.add_column("Runtime (s)", justify="right", style="green")
    table.add_column("Final Cost", justify="right", style="bold")
    table.add_column("Improv.", justify="right")

    for r in results:
        table.add_row(
            str(r["n"]),
            str(r["ga_pop"]),
            str(r["ga_gens"]),
            r["sa_mode"],
            str(r["runtime_sec"]),
            str(r["final_cost"]),
            str(r["improvement"])
        )

    console.print(table)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("experiment_results.csv", index=False)
    console.print("[green]Results saved to experiment_results.csv[/green]")

def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter experiments.")
    parser.add_argument("--n", type=int, nargs="+", required=True, help="Puzzle sizes to test")
    args = parser.parse_args()

    run_experiment(args.n)

if __name__ == "__main__":
    main()
