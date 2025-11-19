#!/usr/bin/env python3
"""
Faster greedy + refinement optimizer with multiprocessing.

Key speedups vs previous version:
- No unary_union in the hot loop.
- For candidates:
    * compute bounding box by per-poly bounds (O(n)), not union.
    * only check overlaps for the new/moved tree (O(n)), not all pairs.
- Use a fast cost (side^2 / n, no overlap area term) during search.
- Only call full layout_cost once per run at the end.

Usage example:

  python3 optimizer_greedy_refine_fast.py \
      --n 80 \
      --runs 8 \
      --workers 4 \
      --samples-two 10000 \
      --samples-new 3000 \
      --refine-sweeps 2 \
      --refine-iters 100
"""

import argparse
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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

from shapely.strtree import STRtree

from geometry import (
    ChristmasTree,
    layout_cost,
    CENTER_BOUNDS,
    scale_factor,
)

console = Console()
SCALE = float(scale_factor)  # 1e15 as float


# ----------------------------------------------------------------------
# Helpers for bounds & overlap
# ----------------------------------------------------------------------


def tree_bounds_float(tree: ChristmasTree) -> Tuple[float, float, float, float]:
    """
    Return (minx, miny, maxx, maxy) in UNscaled float coords.
    """
    minx_s, miny_s, maxx_s, maxy_s = tree.polygon.bounds
    return (
        minx_s / SCALE,
        miny_s / SCALE,
        maxx_s / SCALE,
        maxy_s / SCALE,
    )


def layout_bounds_float(
    trees: List[ChristmasTree],
) -> Tuple[float, float, float, float]:
    """
    Bounds over all trees, UNscaled, no unary_union.
    """
    first = True
    minx = miny = maxx = maxy = 0.0

    for t in trees:
        tx1, ty1, tx2, ty2 = tree_bounds_float(t)
        if first:
            minx, miny, maxx, maxy = tx1, ty1, tx2, ty2
            first = False
        else:
            if tx1 < minx:
                minx = tx1
            if ty1 < miny:
                miny = ty1
            if tx2 > maxx:
                maxx = tx2
            if ty2 > maxy:
                maxy = ty2

    if first:
        return 0.0, 0.0, 0.0, 0.0
    return minx, miny, maxx, maxy


def fast_cost_with_new_tree(
    trees_fixed: List[ChristmasTree],
    candidate: ChristmasTree,
) -> float:
    """
    Fast objective when adding a new tree:
      - reject if candidate overlaps any fixed tree
      - otherwise cost = side^2 / n_new with side from combined bounds
    """
    n_new = len(trees_fixed) + 1

    # Check overlap only for candidate
    fixed_polys = [t.polygon for t in trees_fixed]
    if fixed_polys:
        idx = STRtree(fixed_polys)
        cand_poly = candidate.polygon

        # STRtree.query returns indices, not geometries
        for k in idx.query(cand_poly):
            p = fixed_polys[k]
            if cand_poly.intersects(p) and not cand_poly.touches(p):
                return float("inf")  # hard reject

    # Bounds: combine existing + candidate
    if trees_fixed:
        minx, miny, maxx, maxy = layout_bounds_float(trees_fixed)
        cx1, cy1, cx2, cy2 = tree_bounds_float(candidate)
        minx = min(minx, cx1)
        miny = min(miny, cy1)
        maxx = max(maxx, cx2)
        maxy = max(maxy, cy2)
    else:
        minx, miny, maxx, maxy = tree_bounds_float(candidate)

    width = maxx - minx
    height = maxy - miny
    side = max(width, height)

    return (side * side) / float(n_new)


def fast_cost_full_layout(trees: List[ChristmasTree]) -> float:
    """
    Fast approximate cost for a full layout (no area penalties):
      cost = side^2 / n, where side comes from per-poly bounds.

    We still use real layout_cost at the END for scoring & saving.
    """
    n = len(trees)
    if n == 0:
        return float("inf")

    minx, miny, maxx, maxy = layout_bounds_float(trees)
    side = max(maxx - minx, maxy - miny)
    return (side * side) / float(n)


# ----------------------------------------------------------------------
# Greedy placement (2-tree + incremental)
# ----------------------------------------------------------------------


def best_two_tree_config_fast(
    samples: int,
    rng: random.Random,
) -> Tuple[ChristmasTree, ChristmasTree, float]:
    """
    Fast 2-tree search:
      - Tree1 fixed at (0,0,0)
      - Tree2 random in CENTER_BOUNDS
      - Use fast_cost_with_new_tree
    """
    lo, hi = CENTER_BOUNDS
    t1 = ChristmasTree(center_x="0", center_y="0", angle="0")

    best_cost = float("inf")
    best_t2 = None

    for _ in range(samples):
        x2 = rng.uniform(lo, hi)
        y2 = rng.uniform(lo, hi)
        a2 = rng.uniform(0.0, 360.0)

        t2 = ChristmasTree(center_x=str(x2), center_y=str(y2), angle=str(a2))
        cost = fast_cost_with_new_tree([t1], t2)

        if cost < best_cost:
            best_cost = cost
            best_t2 = t2

    assert best_t2 is not None
    return t1, best_t2, best_cost


def place_new_tree_fast(
    trees_fixed: List[ChristmasTree],
    samples: int,
    rng: random.Random,
) -> Tuple[ChristmasTree, float]:
    """
    Add a new tree to an existing layout using fast incremental cost:
      - sample candidates around current bounds (with padding)
      - reject overlaps
      - minimize side^2 / n
    """
    if not trees_fixed:
        t = ChristmasTree(center_x="0", center_y="0", angle="0")
        return t, fast_cost_full_layout([t])

    minx, miny, maxx, maxy = layout_bounds_float(trees_fixed)
    width = maxx - minx
    height = maxy - miny
    side = max(width, height)
    pad = side if side > 0 else 1.0

    lo_center, hi_center = CENTER_BOUNDS

    minx_f = max(minx - pad, lo_center)
    maxx_f = min(maxx + pad, hi_center)
    miny_f = max(miny - pad, lo_center)
    maxy_f = min(maxy + pad, hi_center)

    best_cost = float("inf")
    best_tree = None

    fixed_polys = [t.polygon for t in trees_fixed]
    idx = STRtree(fixed_polys) if fixed_polys else None

    for _ in range(samples):
        x = rng.uniform(minx_f, maxx_f)
        y = rng.uniform(miny_f, maxy_f)
        a = rng.uniform(0.0, 360.0)

        cand = ChristmasTree(center_x=str(x), center_y=str(y), angle=str(a))
        cand_poly = cand.polygon

        # Overlap check with STRtree if available
        if idx is not None:
            overlaps = False
            for k in idx.query(cand_poly):
                p = fixed_polys[k]
                if cand_poly.intersects(p) and not cand_poly.touches(p):
                    overlaps = True
                    break
            if overlaps:
                continue

        cost = fast_cost_with_new_tree(trees_fixed, cand)
        if cost < best_cost:
            best_cost = cost
            best_tree = cand

    if best_tree is None:
        # If RNG screws us, just put it somewhere and pray
        best_tree = ChristmasTree(center_x="0", center_y="0", angle="0")
        best_cost = fast_cost_with_new_tree(trees_fixed, best_tree)

    return best_tree, best_cost


def build_greedy_layout_fast(
    n: int,
    samples_two: int,
    samples_new: int,
    seed: int,
) -> Tuple[List[ChristmasTree], float]:
    """
    Greedy incremental layout using fast cost.
    """
    rng = random.Random(seed)

    if n <= 0:
        return [], float("inf")
    if n == 1:
        t = ChristmasTree(center_x="0", center_y="0", angle="0")
        return [t], fast_cost_full_layout([t])

    t1, t2, _ = best_two_tree_config_fast(samples_two, rng)
    trees = [t1, t2]

    for _ in range(3, n + 1):
        new_tree, _ = place_new_tree_fast(trees, samples_new, rng)
        trees.append(new_tree)

    return trees, fast_cost_full_layout(trees)


# ----------------------------------------------------------------------
# Local refinement (fast cost)
# ----------------------------------------------------------------------


def refine_single_tree_fast(
    trees: List[ChristmasTree],
    idx: int,
    rng: random.Random,
    step_xy: float,
    step_angle: float,
    iters: int,
) -> float:
    """
    Local hill-climbing for one tree:
      - reject overlaps
      - accept only fast-cost improvements
    """
    n = len(trees)
    if n == 0:
        return float("inf")

    best_layout_cost = fast_cost_full_layout(trees)
    best_tree = trees[idx]

    # Build STRtree for "other" polys (fixed during this local loop)
    other_polys = [t.polygon for j, t in enumerate(trees) if j != idx]
    idx_tree = STRtree(other_polys) if other_polys else None

    for _ in range(iters):
        base = best_tree
        dx = rng.uniform(-step_xy, step_xy)
        dy = rng.uniform(-step_xy, step_xy)
        da = rng.uniform(-step_angle, step_angle)

        new_x = float(base.center_x) + dx
        new_y = float(base.center_y) + dy
        new_a = float(base.angle) + da

        cand = ChristmasTree(center_x=str(new_x), center_y=str(new_y), angle=str(new_a))
        cand_poly = cand.polygon

        # Overlap check vs others
        if idx_tree is not None:
            overlaps = False
            for k in idx_tree.query(cand_poly):
                p = other_polys[k]
                if cand_poly.intersects(p) and not cand_poly.touches(p):
                    overlaps = True
                    break
            if overlaps:
                continue

        # Fast cost: rebuild trees list with candidate at idx
        tmp = list(trees)
        tmp[idx] = cand
        cost = fast_cost_full_layout(tmp)

        if cost < best_layout_cost:
            best_layout_cost = cost
            best_tree = cand

    trees[idx] = best_tree
    return best_layout_cost


def global_refinement_fast(
    trees: List[ChristmasTree],
    sweeps: int,
    refine_iters: int,
    step_xy: float,
    step_angle: float,
    seed: int,
) -> Tuple[List[ChristmasTree], float]:
    rng = random.Random(seed)
    if not trees or sweeps <= 0 or refine_iters <= 0:
        return trees, fast_cost_full_layout(trees)

    for _ in range(sweeps):
        idxs = list(range(len(trees)))
        rng.shuffle(idxs)
        for i in idxs:
            refine_single_tree_fast(
                trees,
                i,
                rng,
                step_xy=step_xy,
                step_angle=step_angle,
                iters=refine_iters,
            )

    return trees, fast_cost_full_layout(trees)


# ----------------------------------------------------------------------
# Worker & saving
# ----------------------------------------------------------------------


def run_single_job(job: Dict[str, Any]) -> Dict[str, Any]:
    run_id = job["run_id"]
    n = job["n"]
    seed = job["seed"]
    samples_two = job["samples_two"]
    samples_new = job["samples_new"]
    sweeps = job["refine_sweeps"]
    refine_iters = job["refine_iters"]
    step_xy = job["step_xy"]
    step_angle = job["step_angle"]

    # Greedy (fast)
    trees, fast_greedy_cost = build_greedy_layout_fast(
        n=n,
        samples_two=samples_two,
        samples_new=samples_new,
        seed=seed,
    )

    # Refinement (fast)
    trees, fast_ref_cost = global_refinement_fast(
        trees,
        sweeps=sweeps,
        refine_iters=refine_iters,
        step_xy=step_xy,
        step_angle=step_angle,
        seed=seed + 100000,
    )

    # Real Kaggle-style cost only once at the end
    true_cost = layout_cost(trees)

    poses = [
        {"x": float(t.center_x), "y": float(t.center_y), "angle": float(t.angle)}
        for t in trees
    ]

    return {
        "run_id": run_id,
        "seed": seed,
        "fast_greedy_cost": fast_greedy_cost,
        "fast_ref_cost": fast_ref_cost,
        "true_cost": true_cost,
        "poses": poses,
    }


def save_layout_kaggle_csv(poses: List[Dict[str, float]], n: int, path: Path) -> None:
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
# Main
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Faster greedy+refinement optimizer with multiprocessing."
    )

    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples-two", type=int, default=10000)
    parser.add_argument("--samples-new", type=int, default=3000)
    parser.add_argument("--refine-sweeps", type=int, default=2)
    parser.add_argument("--refine-iters", type=int, default=100)
    parser.add_argument("--step-xy", type=float, default=0.1)
    parser.add_argument("--step-angle", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", type=str, default="greedy_fast_best")

    args = parser.parse_args()

    n = args.n
    runs = args.runs
    workers = min(args.workers, runs)
    out_path = Path(f"{args.out_prefix}_n{n}.csv")

    console.print(
        f"[bold]Fast greedy+refinement for n={n}[/bold] "
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
        task = progress.add_task(
            description="Running fast greedy-refine jobs",
            total=runs,
        )

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run_single_job, job) for job in jobs]

            for fut in as_completed(futures):
                res = fut.result()
                run_id = res["run_id"]
                seed = res["seed"]
                fg = res["fast_greedy_cost"]
                fr = res["fast_ref_cost"]
                tc = res["true_cost"]

                progress.advance(task, 1)

                console.print(
                    f"[cyan]Run {run_id}[/cyan] (seed={seed}) "
                    f"fast_greedy={fg:.6f}, fast_ref={fr:.6f}, "
                    f"true_cost={tc:.6f}"
                )

                if tc < best_cost:
                    best_cost = tc
                    best_poses = res["poses"]
                    save_layout_kaggle_csv(best_poses, n=n, path=out_path)
                    console.print(
                        f"[green]New best true_cost = {best_cost:.6f}[/green] "
                        f"→ saved to [bold]{out_path}[/bold]"
                    )

    if best_poses is None:
        console.print("[red]No successful runs.[/red]")
        return

    console.print(
        f"[bold green]Done.[/bold green] "
        f"Best true layout_cost for n={n}: {best_cost:.6f} "
        f"(saved in {out_path})"
    )


if __name__ == "__main__":
    main()
