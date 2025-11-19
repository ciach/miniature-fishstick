#!/usr/bin/env python3
"""
x_fit_upside_down.py

One tree upright, one tree upside down.
Slide the second along X (left/right) until they *just* stop overlapping.

- Tree 1: (x=0, y=0), angle = base_angle_deg
- Tree 2: (x=dx, y=0), angle = base_angle_deg + 180

We:
  - binary-search minimal dx > 0 (right side)
  - binary-search minimal dx < 0 (left side, symmetric)
  - report both and the best |dx|
  - plot the best configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from geometry import ChristmasTree, has_overlap
from plot_trees import plot_trees  # adjust import if your file is elsewhere

console = Console()


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------


@dataclass
class SideResult:
    direction: str  # "left" or "right"
    dx: float  # signed dx (negative = left, positive = right)


# ----------------------------------------------------------------------
# Core helpers
# ----------------------------------------------------------------------


def make_pair(dx: float, base_angle_deg: float) -> list[ChristmasTree]:
    """
    Create two trees:
      - t1 at (0, 0), angle = base_angle_deg
      - t2 at (dx, 0), angle = base_angle_deg + 180
    """
    a1 = Decimal(str(base_angle_deg))
    a2 = a1 + Decimal("180")

    t1 = ChristmasTree(center_x=Decimal("0"), center_y=Decimal("0"), angle=a1)
    t2 = ChristmasTree(center_x=Decimal(str(dx)), center_y=Decimal("0"), angle=a2)

    return [t1, t2]


def overlaps(dx: float, base_angle_deg: float) -> bool:
    trees = make_pair(dx, base_angle_deg)
    return has_overlap(trees)


def find_min_dx_one_side(
    base_angle_deg: float,
    direction: str,
    dx_init_hi: float = 2.0,
    dx_max: float = 50.0,
    tol: float = 1e-4,
) -> SideResult | None:
    """
    Binary-search minimal |dx| for given side (left/right) such that there is NO overlap.

    direction: "left"  → dx < 0
               "right" → dx > 0
    """
    sign = -1.0 if direction == "left" else 1.0

    # start at 0: guaranteed overlap
    low = 0.0
    high = dx_init_hi

    # Step 1: find a high such that there is NO overlap
    while True:
        dx_test = sign * high
        if not overlaps(dx_test, base_angle_deg):
            break
        high *= 2.0
        if high > dx_max:
            # no non-overlap found up to dx_max
            return None

    # Step 2: binary search between low (overlap) and high (no-overlap)
    while (high - low) > tol:
        mid = (low + high) / 2.0
        dx_test = sign * mid
        if overlaps(dx_test, base_angle_deg):
            low = mid
        else:
            high = mid

    dx_min = sign * high
    return SideResult(direction=direction, dx=dx_min)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------


def run_search(
    base_angle_deg: float, dx_init_hi: float, dx_max: float, tol: float
) -> tuple[SideResult | None, SideResult | None]:
    """
    Search both sides (left and right) with a simple rich progress.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Searching left/right...", total=2)

        left_res = find_min_dx_one_side(
            base_angle_deg,
            direction="left",
            dx_init_hi=dx_init_hi,
            dx_max=dx_max,
            tol=tol,
        )
        progress.update(task_id, advance=1)

        right_res = find_min_dx_one_side(
            base_angle_deg,
            direction="right",
            dx_init_hi=dx_init_hi,
            dx_max=dx_max,
            tol=tol,
        )
        progress.update(task_id, advance=1)

    return left_res, right_res


def main():
    parser = argparse.ArgumentParser(
        description="Slide an upside-down tree left/right until it fits tightly next to an upright tree."
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=0.0,
        help="Base angle of the first tree in degrees (second is base+180).",
    )
    parser.add_argument("--dx-init-hi", type=float, default=2.0)
    parser.add_argument("--dx-max", type=float, default=50.0)
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args()

    base_angle_deg = args.angle

    left_res, right_res = run_search(
        base_angle_deg=base_angle_deg,
        dx_init_hi=args.dx_init_hi,
        dx_max=args.dx_max,
        tol=args.tol,
    )

    console.print(
        f"\n[bold]Base angle:[/bold] {base_angle_deg}° (second is {base_angle_deg + 180.0}°)"
    )

    if left_res is None and right_res is None:
        console.print("[red]No non-overlapping position found within dx_max.[/red]")
        return

    if left_res is not None:
        console.print(f" Left side:  dx ≈ {left_res.dx:.6f}")
    else:
        console.print(" Left side:  no solution up to dx_max")

    if right_res is not None:
        console.print(f" Right side: dx ≈ {right_res.dx:.6f}")
    else:
        console.print(" Right side: no solution up to dx_max")

    # Pick best by minimal |dx|
    candidates = [r for r in [left_res, right_res] if r is not None]
    best = min(candidates, key=lambda r: abs(r.dx))

    console.print(
        f"\n[bold green]Best tight fit:[/bold green] direction={best.direction}, dx={best.dx:.6f}"
    )

    # Visualize best outcome
    trees = make_pair(best.dx, base_angle_deg)
    title = f"Upright vs upside-down, angle={base_angle_deg}°, dx={best.dx:.6f}"
    out_png = f"x_fit_upside_down_angle{base_angle_deg:.1f}_dx{best.dx:.3f}.png"

    # Adjust args to match your plot_trees() signature if needed
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
