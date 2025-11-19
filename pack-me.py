#!/usr/bin/env python3
"""
Pattern-based Christmas tree packing using your geometry.py

- Uses real ChristmasTree polygons and layout_cost.
- We define a parametric pattern:
    dx: horizontal spacing between trees
    dy: vertical spacing between rows
    offset_fraction: horizontal offset applied to every second row (0..0.5)
    theta_deg: common rotation angle for all trees in the batch

- For given n, we:
    * build a grid of candidate centers around the origin
    * apply staggered rows
    * place trees sequentially, skipping those that would overlap
    * stop when we either place n trees or run out of candidates

- We then evaluate layout_cost(trees) and pick pattern with minimal cost.

This is a heuristic “pattern search” version that matches your idea:
find good dx in x, dy in y, use a simple 2D pattern, and play with angles.
"""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from typing import List

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

from geometry import ChristmasTree, has_overlap, layout_cost
from plot_trees import plot_trees

console = Console()

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

# Reasonable spacing in "unscaled" units, based on tree width ~0.7 and height ~1.0
DX_VALUES = [0.8, 0.9, 1.0, 1.1, 1.2]
DY_VALUES = [0.9, 1.0, 1.1, 1.2]
OFFSET_FRACTIONS = [0.0, 0.25, 0.33, 0.5]
THETA_VALUES = [0.0, 10.0, 20.0, 30.0, 45.0]  # common angle for all trees

N_WORKERS = 8  # adjust to your CPU


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PatternParams:
    dx: float
    dy: float
    offset_fraction: float
    theta_deg: float


@dataclass
class PatternResult:
    params: PatternParams
    count: int
    cost: float
    trees: List[ChristmasTree]


# ----------------------------------------------------------------------
# Pattern construction with real trees
# ----------------------------------------------------------------------


def build_layout_for_n(n: int, params: PatternParams) -> List[ChristmasTree]:
    """
    Build a layout for exactly n trees using a regular/staggered grid pattern.

    - Centers are laid on a grid around origin:
        cx = (col - C/2) * dx + (row%2)*offset_fraction*dx
        cy = (row - R/2) * dy

    - All trees share the same angle theta_deg (batch orientation).

    - Trees are added one by one; each new tree is accepted only if it doesn't
      introduce any overlap (according to has_overlap).

    If we cannot place n trees with this pattern, we return fewer than n.
    """
    dx = params.dx
    dy = params.dy
    off = params.offset_fraction
    theta_deg = params.theta_deg

    if n <= 0:
        return []

    # Rough guess of how many grid cells we need:
    # density ~1 tree per cell, add some slack factor.
    slack_factor = 1.5
    grid_side = int(math.ceil(math.sqrt(n * slack_factor)))
    if grid_side < 1:
        grid_side = 1

    # We'll index rows/cols around 0 to keep layout centered near origin.
    row_indices = list(range(-grid_side // 2, grid_side // 2 + 1))
    col_indices = list(range(-grid_side // 2, grid_side // 2 + 1))

    trees: List[ChristmasTree] = []

    for row in row_indices:
        cy = row * dy
        for col in col_indices:
            cx = col * dx
            if row % 2 != 0:
                cx += off * dx

            # Create tree with given center and common angle
            t = ChristmasTree(
                center_x=Decimal(str(cx)),
                center_y=Decimal(str(cy)),
                angle=Decimal(str(theta_deg)),
            )

            # Try adding to the current list and check for overlap
            candidate_list = trees + [t]
            if has_overlap(candidate_list):
                continue

            trees.append(t)

            if len(trees) >= n:
                return trees

    # Pattern too dense / not enough candidates → fail (caller decides)
    return trees


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------


def evaluate_pattern(args: tuple[int, PatternParams]) -> PatternResult:
    """
    Worker function for multiprocessing.

    args: (n, params)

    Returns:
        PatternResult with:
            - count: how many trees we actually managed to place
            - cost: layout_cost if count == n, else +inf
    """
    n, params = args
    trees = build_layout_for_n(n, params)
    if len(trees) < n:
        # Could not place all trees without overlap using this pattern → invalid
        cost = float("inf")
    else:
        cost = layout_cost(trees)

    return PatternResult(params=params, count=len(trees), cost=cost, trees=trees)


# ----------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------


def build_param_grid() -> list[PatternParams]:
    """Create the grid of parameter combinations to test."""
    grid: list[PatternParams] = []
    for dx, dy, off, theta in product(
        DX_VALUES, DY_VALUES, OFFSET_FRACTIONS, THETA_VALUES
    ):
        # Some basic sanity filters
        if dx <= 0 or dy <= 0:
            continue
        if not (0.0 <= off <= 0.5):
            continue
        grid.append(PatternParams(dx=dx, dy=dy, offset_fraction=off, theta_deg=theta))
    return grid


def run_search(n: int) -> PatternResult | None:
    param_grid = build_param_grid()
    total = len(param_grid)
    console.print(f"[bold]Searching patterns for n={n}[/bold]")
    console.print(f"[bold]Total parameter combinations:[/bold] {total}")

    best: PatternResult | None = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Evaluating patterns...", total=total)

        work_items = [(n, p) for p in param_grid]

        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {
                executor.submit(evaluate_pattern, args): args for args in work_items
            }

            for fut in as_completed(futures):
                res: PatternResult = fut.result()
                progress.update(task_id, advance=1)

                if res.cost < float("inf"):
                    if best is None or res.cost < best.cost:
                        best = res

    return best


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main():
    global N_WORKERS
    parser = argparse.ArgumentParser(
        description="Pattern-based Christmas tree layout search (using geometry.py)."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=80,
        help="Number of trees to place.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=N_WORKERS,
        help="Number of worker processes.",
    )
    args = parser.parse_args()

    N_WORKERS = max(1, args.workers)

    best = run_search(args.n)

    if best is None:
        console.print(
            "[red]No valid pattern found (all patterns failed to place n trees).[/red]"
        )
        return

    p = best.params
    console.print("\n[green]Best pattern found:[/green]")
    console.print(
        f"  dx = {p.dx:.3f}, dy = {p.dy:.3f}, "
        f"offset_fraction = {p.offset_fraction:.3f}, "
        f"theta = {p.theta_deg:.1f}°"
    )
    console.print(f"  Placed trees: {best.count}")
    console.print(f"  layout_cost: [bold]{best.cost:.9f}[/bold]")

    # Visualize best layout for quick sanity check
    if best is not None:
        title = f"Pattern layout n={args.n}, cost={best.cost:.6f}"
        out_png = f"pattern_n{args.n}.png"
        plot_trees(
            best.trees, title=title, show_square=True, padding=0.5, output_path=out_png
        )
        console.print(f"Saved plot to {out_png}")


if __name__ == "__main__":
    main()
