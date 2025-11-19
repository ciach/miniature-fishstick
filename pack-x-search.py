#!/usr/bin/env python3
"""
pack_x_search.py

Explore minimal horizontal spacing (dx) between TWO Christmas trees.

- Tree 1 at (0, 0)
- Tree 2 at (dx, 0)
- Both share the same angle theta (in degrees)

For each theta, we binary-search the smallest dx > 0 such that
the two trees do NOT overlap according to geometry.has_overlap.

At the end, we also PLOT the best pair using plot_trees.
"""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
from typing import List

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

from geometry import ChristmasTree, has_overlap, bounding_square_side
from plot_trees import plot_trees  # <-- make sure this exists in your repo

console = Console()


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass
class XPairResult:
    theta_deg: float
    dx_min: float
    side_length: float  # bounding square side for the 2-tree layout


# ----------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------


def overlap_for_dx(theta_deg: float, dx: Decimal) -> bool:
    """
    Build two trees with given angle and dx and check overlap.

    Tree 1: center_x=0, center_y=0
    Tree 2: center_x=dx, center_y=0
    Both share angle=theta_deg.
    """
    angle_dec = Decimal(str(theta_deg))
    t1 = ChristmasTree(center_x=Decimal("0"), center_y=Decimal("0"), angle=angle_dec)
    t2 = ChristmasTree(center_x=dx, center_y=Decimal("0"), angle=angle_dec)
    return has_overlap([t1, t2])


def find_min_dx_for_angle(
    theta_deg: float, dx_init_hi: float = 2.0, dx_max: float = 50.0, tol: float = 1e-4
) -> XPairResult | None:
    """
    For a fixed angle, find the minimal dx such that two trees do NOT overlap.

    Strategy:
        - low = 0  (definitely overlapping, same center)
        - high starts at dx_init_hi
        - increase high until we find a non-overlapping placement or hit dx_max
        - then binary search between low and high to tolerance `tol`

    Returns:
        XPairResult or None if we failed to find a non-overlapping dx up to dx_max.
    """
    low = Decimal("0")
    high = Decimal(str(dx_init_hi))
    dx_max_dec = Decimal(str(dx_max))
    tol_dec = Decimal(str(tol))

    # ensure there's at least one non-overlap at some high
    while True:
        if not overlap_for_dx(theta_deg, high):
            break
        high *= 2
        if high > dx_max_dec:
            return None

    # binary search between low (overlap) and high (no overlap)
    while (high - low) > tol_dec:
        mid = (low + high) / Decimal("2")
        if overlap_for_dx(theta_deg, mid):
            low = mid
        else:
            high = mid

    dx_min = float(high)

    # build final 2-tree layout and measure bounding side
    t1 = ChristmasTree(center_x="0", center_y="0", angle=str(theta_deg))
    t2 = ChristmasTree(center_x=str(dx_min), center_y="0", angle=str(theta_deg))
    side, _ = bounding_square_side([t1, t2])
    side_f = float(side)

    return XPairResult(theta_deg=theta_deg, dx_min=dx_min, side_length=side_f)


# ----------------------------------------------------------------------
# Multiprocessing wrapper
# ----------------------------------------------------------------------


def _worker_angle_search(
    theta_deg: float, dx_init_hi: float, dx_max: float, tol: float
) -> XPairResult | None:
    return find_min_dx_for_angle(
        theta_deg, dx_init_hi=dx_init_hi, dx_max=dx_max, tol=tol
    )


# ----------------------------------------------------------------------
# Main orchestration
# ----------------------------------------------------------------------


def run_angle_scan(
    theta_min: float = 0.0,
    theta_max: float = 90.0,
    theta_step: float = 1.0,
    dx_init_hi: float = 2.0,
    dx_max: float = 50.0,
    tol: float = 1e-4,
    workers: int = 4,
) -> List[XPairResult]:
    """
    Scan angles in [theta_min, theta_max] with step theta_step and compute
    minimal dx for each, using multiprocessing.
    """
    angles: List[float] = []
    theta = theta_min
    while theta <= theta_max + 1e-9:
        angles.append(round(theta, 6))
        theta += theta_step

    console.print(
        f"[bold]Scanning minimal dx for angles from {theta_min}° to {theta_max}° "
        f"step {theta_step}°[/bold]"
    )
    console.print(f"Total angles: {len(angles)}")

    results: List[XPairResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Evaluating angles...", total=len(angles))

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _worker_angle_search,
                    theta,
                    dx_init_hi,
                    dx_max,
                    tol,
                ): theta
                for theta in angles
            }

            for fut in as_completed(futures):
                res = fut.result()
                progress.update(task_id, advance=1)
                if res is not None:
                    results.append(res)

    results = [r for r in results if r is not None]
    results.sort(key=lambda r: r.dx_min)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Explore minimal horizontal spacing between two trees as a function of angle."
    )
    parser.add_argument("--theta-min", type=float, default=0.0)
    parser.add_argument("--theta-max", type=float, default=90.0)
    parser.add_argument("--theta-step", type=float, default=1.0)
    parser.add_argument("--dx-init-hi", type=float, default=2.0)
    parser.add_argument("--dx-max", type=float, default=50.0)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many best angles to print.",
    )

    args = parser.parse_args()

    results = run_angle_scan(
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        theta_step=args.theta_step,
        dx_init_hi=args.dx_init_hi,
        dx_max=args.dx_max,
        tol=args.tol,
        workers=args.workers,
    )

    if not results:
        console.print("[red]No valid dx found for any angle in given range.[/red]")
        return

    k = min(args.top_k, len(results))

    console.print(
        "\n[bold green]Best angles by minimal dx (tightest in X):[/bold green]"
    )
    for r in results[:k]:
        console.print(
            f"θ = {r.theta_deg:6.2f}°  |  dx_min = {r.dx_min:8.5f}  |  side(2 trees) = {r.side_length:8.5f}"
        )

    best = results[0]
    console.print(
        f"\n[bold]Absolute best in this scan:[/bold] "
        f"θ = {best.theta_deg:.3f}°, dx_min = {best.dx_min:.6f}, side = {best.side_length:.6f}"
    )

    # -------- visualize the best pair --------
    angle_str = f"{best.theta_deg:.3f}"
    dx_str = f"{best.dx_min:.6f}"

    t1 = ChristmasTree(center_x="0", center_y="0", angle=angle_str)
    t2 = ChristmasTree(center_x=dx_str, center_y="0", angle=angle_str)
    trees = [t1, t2]

    title = f"Best X-pair: θ={angle_str}°, dx={dx_str}"
    output_path = f"best_x_pair_theta{best.theta_deg:.1f}_dx{best.dx_min:.3f}.png"

    # Adjust kwargs to match your plot_trees signature
    plot_trees(
        trees,
        title=title,
        show_square=True,
        padding=0.5,
        output_path=output_path,
    )

    console.print(f"\n[bold green]Saved best pair plot to[/bold green] {output_path}")


if __name__ == "__main__":
    main()
