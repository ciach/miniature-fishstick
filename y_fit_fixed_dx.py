#!/usr/bin/env python3
"""
y_fit_fixed_dx.py

Given:
  - one tree at (0, 0) with angle = base_angle_deg
  - second tree at (dx_fixed, dy) with angle = base_angle_deg + 180 (flipped)

We keep dx_fixed constant and slide the second tree vertically (change dy)
until they *just* stop overlapping.

We:
  - binary-search minimal |dy| > 0 where has_overlap() is False
  - allow searching "down" (dy < 0) or "up" (dy > 0) or both
  - pick the best (smallest |dy|)
  - plot the resulting pair.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from geometry import ChristmasTree, has_overlap
from plot_trees import plot_trees  # adjust import if needed

console = Console()


# ----------------------------------------------------------------------
# Data structure
# ----------------------------------------------------------------------


@dataclass
class YSideResult:
    direction: str  # "up" or "down"
    dy: float  # signed dy (positive = up, negative = down)


# ----------------------------------------------------------------------
# Core helpers
# ----------------------------------------------------------------------


def make_pair(dx: float, dy: float, base_angle_deg: float) -> list[ChristmasTree]:
    """
    Create two trees:
      - t1 at (0, 0), angle = base_angle_deg
      - t2 at (dx, dy), angle = base_angle_deg + 180
    """
    a1 = Decimal(str(base_angle_deg))
    a2 = a1 + Decimal("180")

    t1 = ChristmasTree(center_x=Decimal("0"), center_y=Decimal("0"), angle=a1)
    t2 = ChristmasTree(center_x=Decimal(str(dx)), center_y=Decimal(str(dy)), angle=a2)

    return [t1, t2]


def overlaps(dx: float, dy: float, base_angle_deg: float) -> bool:
    trees = make_pair(dx, dy, base_angle_deg)
    return has_overlap(trees)


def find_min_dy_one_side(
    dx_fixed: float,
    base_angle_deg: float,
    direction: str,
    dy_init_hi: float = 2.0,
    dy_max: float = 50.0,
    tol: float = 1e-4,
) -> YSideResult | None:
    """
    Binary search minimal |dy| for a given vertical direction ("up" / "down")
    such that there is NO overlap.

    direction = "up"   -> dy > 0
    direction = "down" -> dy < 0
    """
    sign = 1.0 if direction == "up" else -1.0

    low = 0.0
    high = dy_init_hi

    # Step 1: find some high where there's no overlap
    while True:
        dy_test = sign * high
        if not overlaps(dx_fixed, dy_test, base_angle_deg):
            break
        high *= 2.0
        if high > dy_max:
            # couldn't separate them within dy_max
            return None

    # Step 2: binary search between low (overlap) and high (no-overlap)
    while (high - low) > tol:
        mid = (low + high) / 2.0
        dy_test = sign * mid
        if overlaps(dx_fixed, dy_test, base_angle_deg):
            low = mid
        else:
            high = mid

    dy_min = sign * high
    return YSideResult(direction=direction, dy=dy_min)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------


def run_search(
    dx_fixed: float,
    base_angle_deg: float,
    dy_init_hi: float,
    dy_max: float,
    tol: float,
    both_sides: bool = True,
) -> tuple[YSideResult | None, YSideResult | None]:
    """
    Run search for 'down' and 'up' (or only down, if you want) with a tiny progress.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        todo = 2 if both_sides else 1
        task_id = progress.add_task("Searching vertical fit...", total=todo)

        down_res = find_min_dy_one_side(
            dx_fixed=dx_fixed,
            base_angle_deg=base_angle_deg,
            direction="down",
            dy_init_hi=dy_init_hi,
            dy_max=dy_max,
            tol=tol,
        )
        progress.update(task_id, advance=1)

        if both_sides:
            up_res = find_min_dy_one_side(
                dx_fixed=dx_fixed,
                base_angle_deg=base_angle_deg,
                direction="up",
                dy_init_hi=dy_init_hi,
                dy_max=dy_max,
                tol=tol,
            )
            progress.update(task_id, advance=1)
        else:
            up_res = None

    return down_res, up_res


def main():
    parser = argparse.ArgumentParser(
        description="Slide an upside-down tree vertically relative to an upright one until they fit tightly."
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=0.0,
        help="Base angle of the first tree (second is base+180).",
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=0.0,
        help="Fixed horizontal offset between trees.",
    )
    parser.add_argument("--dy-init-hi", type=float, default=2.0)
    parser.add_argument("--dy-max", type=float, default=50.0)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument(
        "--only-down",
        action="store_true",
        help="Only search with the top tree moving down (negative dy).",
    )
    args = parser.parse_args()

    base_angle_deg = args.angle
    dx_fixed = args.dx

    console.print(
        f"\n[bold]Setup:[/bold] base_angle={base_angle_deg}°, "
        f"second angle={base_angle_deg + 180.0}°, dx_fixed={dx_fixed}"
    )

    down_res, up_res = run_search(
        dx_fixed=dx_fixed,
        base_angle_deg=base_angle_deg,
        dy_init_hi=args.dy_init_hi,
        dy_max=args.dy_max,
        tol=args.tol,
        both_sides=not args.only_down,
    )

    if down_res is None and (up_res is None or args.only_down):
        console.print(
            "[red]No non-overlapping vertical position found within dy_max.[/red]"
        )
        return

    if down_res is not None:
        console.print(f" Down: dy ≈ {down_res.dy:.6f}")
    else:
        console.print(" Down: no solution up to dy_max")

    if up_res is not None:
        console.print(f" Up:   dy ≈ {up_res.dy:.6f}")

    # Pick best by minimal |dy|
    candidates = [r for r in [down_res, up_res] if r is not None]
    best = min(candidates, key=lambda r: abs(r.dy))

    console.print(
        f"\n[bold green]Best tight vertical fit:[/bold green] "
        f"direction={best.direction}, dy={best.dy:.6f}"
    )

    # Visualize
    trees = make_pair(dx_fixed, best.dy, base_angle_deg)
    title = (
        f"Vertical fit at fixed dx={dx_fixed}, "
        f"angle={base_angle_deg}° / {base_angle_deg+180.0}°, dy={best.dy:.6f}"
    )
    out_png = f"y_fit_dx{dx_fixed:.3f}_angle{base_angle_deg:.1f}_dy{best.dy:.3f}.png"

    plot_trees(
        trees,
        title=title,
        show_square=True,
        padding=0.5,
        output_path=out_png,
    )

    console.print(f"\nSaved plot to [bold]{out_png}[/bold]")


if __name__ == "__main__":
    main()
