#!/usr/bin/env python3
"""
Greedy + refinement optimizer with multiprocessing.

What it does:
- For a fixed n:
  - Run multiple greedy builds in parallel (different seeds).
  - Each build:
      * Greedy incremental placement (trees added one by one).
      * Global local-refinement sweep over all trees.
  - The main process tracks the best layout and saves it to a Kaggle-style CSV
    that you can plot with plot_from_csv.py.

Usage example:

  python3 optimizer_greedy_refine.py --n 80 --runs 8 --workers 4 \
      --samples-two 40000 --samples-new 8000 --refine-iters 200

Then to plot:

  python3 plot_from_csv.py --csv greedy_best_n80.csv --n 80
"""

import argparse
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from geometry import (
    ChristmasTree,
    layout_cost,
    bounding_square_side,
    CENTER_BOUNDS,
    placed_polygons,
    scale_factor,
)

console = Console()


# ----------------------------------------------------------------------
# Serialization helpers (for passing results between processes)
# ----------------------------------------------------------------------


@dataclass
class TreePose:
    x: float
    y: float
    angle: float


def trees_to_poses(trees: List[ChristmasTree]) -> List[TreePose]:
    return [
        TreePose(float(t.center_x), float(t.center_y), float(t.angle)) for t in trees
    ]


def poses_to_trees(poses: List[TreePose]) -> List[ChristmasTree]:
    trees: List[ChristmasTree] = []
    for p in poses:
        t = ChristmasTree(center_x=str(p.x), center_y=str(p.y), angle=str(p.angle))
        trees.append(t)
    return trees


# ----------------------------------------------------------------------
# Greedy placement helpers
# ----------------------------------------------------------------------


def best_two_tree_config(
    samples: int,
    rng: random.Random,
) -> Tuple[ChristmasTree, ChristmasTree, float]:
    """
    Find a decent 2-tree configuration:
      - Tree 1 fixed at (0, 0, 0)
      - Tree 2 sampled randomly in CENTER_BOUNDS
    using layout_cost as objective.
    """
    lo, hi = CENTER_BOUNDS

    t1 = ChristmasTree(center_x="0", center_y="0", angle="0")

    best_cost = float("inf")
    best_t2 = None

    for _ in range(samples):
        x2 = rng.uniform(lo, hi)
        y2 = rng.uniform(lo, hi)
        angle2 = rng.uniform(0.0, 360.0)

        t2 = ChristmasTree(
            center_x=str(x2),
            center_y=str(y2),
            angle=str(angle2),
        )

        cost = layout_cost([t1, t2])
        if cost < best_cost:
            best_cost = cost
            best_t2 = t2

    assert best_t2 is not None
    return t1, best_t2, best_cost


def place_new_tree(
    trees_fixed: List[ChristmasTree],
    samples: int,
    rng: random.Random,
) -> Tuple[ChristmasTree, float]:
    """
    Given a fixed list of trees, place a new tree greedily:

    - Get current bounding square.
    - Define a padded search region around it.
    - Sample random (x, y, angle) in that region.
    - Choose the pose giving minimal layout_cost.

    This is still "dumb but effective", just targeted around existing trees.
    """
    if not trees_fixed:
        new_tree = ChristmasTree(center_x="0", center_y="0", angle="0")
        return new_tree, layout_cost([new_tree])

    side, (minx, miny, maxx, maxy) = bounding_square_side(trees_fixed)
    pad = side if side > Decimal("0") else Decimal("1.0")

    lo_center, hi_center = CENTER_BOUNDS

    minx_f = max(float(minx - pad), lo_center)
    maxx_f = min(float(maxx + pad), hi_center)
    miny_f = max(float(miny - pad), lo_center)
    maxy_f = min(float(maxy + pad), hi_center)

    best_cost = float("inf")
    best_tree = None

    for _ in range(samples):
        x = rng.uniform(minx_f, maxx_f)
        y = rng.uniform(miny_f, maxy_f)
        angle = rng.uniform(0.0, 360.0)

        candidate = ChristmasTree(
            center_x=str(x),
            center_y=str(y),
            angle=str(angle),
        )

        cost = layout_cost(trees_fixed + [candidate])
        if cost < best_cost:
            best_cost = cost
            best_tree = candidate

    assert best_tree is not None
    return best_tree, best_cost


def build_greedy_layout(
    n: int,
    samples_two: int,
    samples_new: int,
    seed: int,
) -> Tuple[List[ChristmasTree], float]:
    """
    Build a layout for n trees using the incremental greedy approach:
      - For n == 1: single tree at the origin.
      - For n >= 2: best 2-tree config, then add trees one by one.
    """
    rng = random.Random(seed)

    if n <= 0:
        return [], float("inf")

    if n == 1:
        t = ChristmasTree(center_x="0", center_y="0", angle="0")
        return [t], layout_cost([t])

    # 1) Two-tree configuration
    t1, t2, _ = best_two_tree_config(samples_two, rng)
    trees: List[ChristmasTree] = [t1, t2]

    # 2) Add each new tree greedily
    for _ in range(3, n + 1):
        new_tree, _ = place_new_tree(trees, samples_new, rng)
        trees.append(new_tree)

    final_cost = layout_cost(trees)
    return trees, final_cost


# ----------------------------------------------------------------------
# Local refinement
# ----------------------------------------------------------------------


def refine_single_tree(
    trees: List[ChristmasTree],
    idx: int,
    rng: random.Random,
    step_xy: float,
    step_angle: float,
    iters: int,
) -> float:
    """
    Random local search / hill climbing for a single tree while others are frozen.

    - Start from current pose.
    - Try random small moves.
    - Keep only moves that improve layout_cost.
    """
    n = len(trees)
    if n == 0:
        return float("inf")

    base_tree = trees[idx]
    best_tree = base_tree
    best_cost = layout_cost(trees)

    for _ in range(iters):
        dx = rng.uniform(-step_xy, step_xy)
        dy = rng.uniform(-step_xy, step_xy)
        da = rng.uniform(-step_angle, step_angle)

        new_x = float(best_tree.center_x) + dx
        new_y = float(best_tree.center_y) + dy
        new_angle = float(best_tree.angle) + da

        candidate = ChristmasTree(
            center_x=str(new_x),
            center_y=str(new_y),
            angle=str(new_angle),
        )

        new_trees = list(trees)
        new_trees[idx] = candidate

        cost = layout_cost(new_trees)
        if cost < best_cost:
            best_cost = cost
            best_tree = candidate

    trees[idx] = best_tree
    return best_cost


def global_refinement(
    trees: List[ChristmasTree],
    sweeps: int,
    refine_iters: int,
    step_xy: float,
    step_angle: float,
    seed: int,
) -> Tuple[List[ChristmasTree], float]:
    """
    Perform several refinement sweeps over all trees:
      - For each sweep, iterate over all trees,
        and locally improve them one-by-one.
    """
    rng = random.Random(seed)
    if not trees or sweeps <= 0 or refine_iters <= 0:
        return trees, layout_cost(trees)

    for _ in range(sweeps):
        indices = list(range(len(trees)))
        rng.shuffle(indices)  # random order each sweep

        for idx in indices:
            refine_single_tree(
                trees,
                idx,
                rng=rng,
                step_xy=step_xy,
                step_angle=step_angle,
                iters=refine_iters,
            )

    final_cost = layout_cost(trees)
    return trees, final_cost


# ----------------------------------------------------------------------
# Worker for multiprocessing
# ----------------------------------------------------------------------


def run_single_greedy_job(job_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    One job = one multi-start run:
      - Greedy build
      - Global refinement
    Returns:
      {
        "run_id": int,
        "seed": int,
        "cost": float,
        "poses": List[TreePose-like dicts]
      }
    """
    run_id = job_args["run_id"]
    n = job_args["n"]
    seed = job_args["seed"]
    samples_two = job_args["samples_two"]
    samples_new = job_args["samples_new"]
    sweeps = job_args["refine_sweeps"]
    refine_iters = job_args["refine_iters"]
    step_xy = job_args["step_xy"]
    step_angle = job_args["step_angle"]

    # Greedy layout
    trees, greedy_cost = build_greedy_layout(
        n=n,
        samples_two=samples_two,
        samples_new=samples_new,
        seed=seed,
    )

    # Refinement
    trees, refined_cost = global_refinement(
        trees,
        sweeps=sweeps,
        refine_iters=refine_iters,
        step_xy=step_xy,
        step_angle=step_angle,
        seed=seed + 100000,  # different RNG stream
    )

    poses = [
        {"x": float(t.center_x), "y": float(t.center_y), "angle": float(t.angle)}
        for t in trees
    ]

    return {
        "run_id": run_id,
        "seed": seed,
        "cost": refined_cost,
        "poses": poses,
        "greedy_cost": greedy_cost,
    }


# ----------------------------------------------------------------------
# Saving / output
# ----------------------------------------------------------------------


def save_layout_kaggle_csv(
    poses: List[Dict[str, float]],
    n: int,
    path: Path,
) -> None:
    """
    Save a single-n layout in Kaggle-style CSV that works with plot_from_csv.py:

      id,x,y,deg
      010_0,s0.123456,s-0.234567,s45.000000
      ...

    """
    rows = []
    for idx, p in enumerate(poses):
        rows.append(
            {
                "id": f"{n:03d}_{idx}",
                "x": "s" + f"{p['x']:.6f}",
                "y": "s" + f"{p['y']:.6f}",
                "deg": "s" + f"{p['angle']:.6f}",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


# ----------------------------------------------------------------------
# Main CLI logic
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedy + refinement optimizer with multiprocessing."
    )

    parser.add_argument("--n", type=int, required=True, help="Number of trees.")
    parser.add_argument(
        "--runs",
        type=int,
        default=8,
        help="Number of independent runs (multi-start).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--samples-two",
        type=int,
        default=20000,
        help="Random samples for initial 2-tree search per run.",
    )
    parser.add_argument(
        "--samples-new",
        type=int,
        default=5000,
        help="Random samples for each new tree per run.",
    )
    parser.add_argument(
        "--refine-sweeps",
        type=int,
        default=2,
        help="Number of global refinement sweeps per run.",
    )
    parser.add_argument(
        "--refine-iters",
        type=int,
        default=200,
        help="Local refinement iterations per tree per sweep.",
    )
    parser.add_argument(
        "--step-xy",
        type=float,
        default=0.1,
        help="Max XY step size in local refinement.",
    )
    parser.add_argument(
        "--step-angle",
        type=float,
        default=5.0,
        help="Max angle step (degrees) in local refinement.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for all runs.",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="greedy_best",
        help="Prefix for best layout CSV file.",
    )

    args = parser.parse_args()

    n = args.n
    runs = args.runs
    workers = min(args.workers, runs)

    out_path = Path(f"{args.out_prefix}_n{n}.csv")

    console.print(
        f"[bold]Greedy + refinement for n={n}[/bold] "
        f"(runs={runs}, workers={workers})"
    )

    jobs = []
    for run_id in range(runs):
        jobs.append(
            {
                "run_id": run_id,
                "n": n,
                "seed": args.seed + run_id,
                "samples_two": args.samples_two,
                "samples_new": args.samples_new,
                "refine_sweeps": args.refine_sweeps,
                "refine_iters": args.refine_iters,
                "step_xy": args.step_xy,
                "step_angle": args.step_angle,
            }
        )

    best_cost = float("inf")
    best_poses: List[Dict[str, float]] | None = None

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        task_runs = progress.add_task(
            description="Running greedy-refine jobs",
            total=runs,
        )

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run_single_greedy_job, job) for job in jobs]

            for fut in as_completed(futures):
                result = fut.result()

                run_id = result["run_id"]
                seed = result["seed"]
                cost = result["cost"]
                greedy_cost = result["greedy_cost"]

                progress.advance(task_runs, 1)

                console.print(
                    f"[cyan]Run {run_id}[/cyan] "
                    f"(seed={seed}) "
                    f"greedy_cost={greedy_cost:.6f}, "
                    f"refined_cost={cost:.6f}"
                )

                if cost < best_cost:
                    best_cost = cost
                    best_poses = result["poses"]
                    save_layout_kaggle_csv(best_poses, n=n, path=out_path)
                    console.print(
                        f"[green]New best cost = {best_cost:.6f}[/green] "
                        f"→ saved to [bold]{out_path}[/bold]"
                    )

    if best_poses is None:
        console.print("[red]No successful runs. Something broke.[/red]")
        return

    console.print(
        f"[bold green]Done.[/bold green] Best cost for n={n}: {best_cost:.6f} "
        f"(saved in {out_path})"
    )


if __name__ == "__main__":
    main()
