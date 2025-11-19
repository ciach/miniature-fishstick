#!/usr/bin/env python3
"""
pack_x_search_flip.py

Explore minimal horizontal spacing (dx) between TWO Christmas trees where:
- Tree 1 has angle = theta
- Tree 2 has angle = theta + 180 (upside down)

Both are on the same horizontal line:
    Tree 1 at (0, 0)
    Tree 2 at (dx, 0)

For each theta, we binary-search the smallest dx > 0 such that they do NOT
overlap according to geometry.has_overlap.

At the end, we PLOT the best pair using plot_trees.
"""

from __future__ import annotations

import argparse
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
from plot_trees import plot_trees

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

    Tree 1: center_x=0, center_y=0, angle = theta_deg
    Tree 2: center_x=dx, center_y=0, angle = theta_deg + 180 (upside down)
    """
    angle1 = Decimal(str(theta_deg))
    angle2 = angle1 + Decimal("180")

    t1 = ChristmasTree(center_x=Decimal("0"), center_y=Decimal("0"), angle=angle1)
    t2 = ChristmasTree(center_x=dx, center_y=Decimal("0"), angle=angle2)

    return has_overlap([t1, t2])


def find_min_dx_for_angle(
    theta_deg: float,
    dx_init_hi: float = 2.0,
    dx_max: float = 50.0,
    tol: float = 1e-4,
) -> XPairResult | None:
    """
    For a fixed angle, find the minimal dx such that two trees do NOT overlap.

    Strategy:
      - low = 0 (definitely overlapping, same center)
      - high starts at dx_init_hi
      - increase high until no-overlap or dx_max
      - binary search between low / high to tolerance tol
    """
    low = Decimal("0")
    high = Decimal(str(dx_init_hi))
    dx_max_dec = Decimal(str(dx_max))
    tol_dec = Decimal(str(tol))

    # Step 1: ensure we have a "no overlap" high
    while True:
        if not overlap_for_dx(theta_deg, high):
            break
        high *= 2
        if high > dx_max_dec:
            # Can't find a distance up to dx_max with no overlap
            return None

    # Step 2: binary search
    while (high - low) > tol_dec:
        mid = (low + high) / Decimal("2")
        if overlap_for_dx(theta_deg, mid):
            low = mid
        else:
            high = mid

    dx_min = float(high)

    # Build final layout and measure bounding square side
    angle1 = Decimal(str(theta_deg))
    angle2 = angle1 + Decimal("180")

    t1 = ChristmasTree(center_x="0", center_y="0", angle=angle1)
    t2 = ChristmasTree(center_x=str(dx_min), center_y="0", angle=angle2)
    side, _ = bounding_square_side([t1, t2])
    side_f = float(side)

    return XPairResult(theta_deg=theta_deg, dx_min=dx_min, side_length=side_f)


# ----------------------------------------------------------------------
# Multiprocessing wrapper
# ----------------------------------------------------------------------


def _worker_angle_search(
    theta_deg: float,
    dx_init_hi: float,
    dx_max: float,
    tol: float,
) -> XPairResult | None:
    return find_min_dx_for_angle(
        theta_deg, dx_init_hi=dx_init_hi, dx_max=dx_max, tol=tol
    )


# ----------------------------------------------------------------------
# Scan angles
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
        f"[bold]Scanning minimal dx for flipped pair from {theta_min}° to {theta_max}° "
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

    # Filter out failures and sort by dx_min
    results = [r for r in results if r is not None]
    results.sort(key=lambda r: r.dx_min)
    return results


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Minimal horizontal spacing between two trees with second flipped 180°."
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
        "\n[bold green]Best angles by minimal dx (second flipped):[/bold green]"
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
    angle1 = Decimal(str(best.theta_deg))
    angle2 = angle1 + Decimal("180")
    dx_str = f"{best.dx_min:.6f}"

    t1 = ChristmasTree(center_x="0", center_y="0", angle=angle1)
    t2 = ChristmasTree(center_x=dx_str, center_y="0", angle=angle2)
    trees = [t1, t2]

    title = f"Best flipped X-pair: θ={best.theta_deg:.3f}° / θ+180°, dx={dx_str}"
    output_path = (
        f"best_x_pair_flipped_theta{best.theta_deg:.1f}_dx{best.dx_min:.3f}.png"
    )

    plot_trees(
        trees,
        title=title,
        show_square=True,
        padding=0.5,
        output_path=output_path,
    )

    console.print(
        f"\n[bold green]Saved best flipped pair plot to[/bold green] {output_path}"
    )


if __name__ == "__main__":
    main()
