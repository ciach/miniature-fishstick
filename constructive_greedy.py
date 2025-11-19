#!/usr/bin/env python3
"""
Greedy constructive layout:
- First find a good placement for 2 trees (using Kaggle layout_cost).
- Freeze them.
- Then add trees one by one, always minimizing layout_cost w.r.t. the new tree only.

Usage:
    python3 constructive_greedy.py --n 80
"""

import argparse
import csv
import math
import random
from typing import List, Tuple

from decimal import Decimal

from geometry import (
    ChristmasTree,
    layout_cost,
    bounding_square_side,
    CENTER_BOUNDS,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def best_two_tree_config(
    samples: int,
    rng: random.Random,
) -> Tuple[ChristmasTree, ChristmasTree, float]:
    """
    Find a good 2-tree configuration:
    - Tree 1 fixed at (0,0,0)
    - Tree 2 sampled randomly in CENTER_BOUNDS
    """
    lo, hi = CENTER_BOUNDS

    t1 = ChristmasTree(center_x="0", center_y="0", angle="0")

    best_cost = float("inf")
    best_t2: ChristmasTree | None = None

    for i in range(samples):
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
    Given a fixed list of already placed trees, find a good position
    for a *new* tree (same shape), minimizing layout_cost.

    We:
      - Compute current bounding square.
      - Sample candidate positions in a padded box around existing trees.
    """
    if not trees_fixed:
        # Shouldn't happen in this flow, but handle anyway.
        new_tree = ChristmasTree(center_x="0", center_y="0", angle="0")
        return new_tree, layout_cost([new_tree])

    # Get current bounding square for existing layout
    side, (minx, miny, maxx, maxy) = bounding_square_side(trees_fixed)
    pad = side if side > Decimal("0") else Decimal("1.0")

    lo_center, hi_center = CENTER_BOUNDS

    # Restrict search region, but keep it inside Kaggle bounds
    minx_f = max(float(minx - pad), lo_center)
    maxx_f = min(float(maxx + pad), hi_center)
    miny_f = max(float(miny - pad), lo_center)
    maxy_f = min(float(maxy + pad), hi_center)

    best_cost = float("inf")
    best_tree: ChristmasTree | None = None

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
    Build a layout for n trees, greedily:
      - For n == 1: single tree at origin.
      - For n >= 2: best 2-tree config, then add trees one by one.
    """
    rng = random.Random(seed)

    if n <= 0:
        return [], float("inf")

    # n == 1: trivial
    if n == 1:
        t = ChristmasTree(center_x="0", center_y="0", angle="0")
        return [t], layout_cost([t])

    # Step 1: find a good 2-tree configuration
    t1, t2, cost_2 = best_two_tree_config(samples_two, rng)
    trees: List[ChristmasTree] = [t1, t2]

    # Step 2+: add further trees greedily
    for k in range(3, n + 1):
        new_tree, _ = place_new_tree(trees, samples_new, rng)
        trees.append(new_tree)

    final_cost = layout_cost(trees)
    return trees, final_cost


def save_layout_to_csv(trees: List[ChristmasTree], path: str) -> None:
    """
    Save layout in Kaggle format:
        id,center_x,center_y,angle
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "center_x", "center_y", "angle"])
        for idx, t in enumerate(trees):
            writer.writerow(
                [
                    idx,
                    str(t.center_x),
                    str(t.center_y),
                    str(t.angle),
                ]
            )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedy constructive tree placement (incremental)."
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Number of trees to place.",
    )
    parser.add_argument(
        "--samples-two",
        type=int,
        default=20000,
        help="Number of random samples for the initial 2-tree search.",
    )
    parser.add_argument(
        "--samples-new",
        type=int,
        default=5000,
        help="Number of random samples for each newly added tree.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    trees, final_cost = build_greedy_layout(
        n=args.n,
        samples_two=args.samples_two,
        samples_new=args.samples_new,
        seed=args.seed,
    )

    print(f"Placed {len(trees)} trees.")
    print(f"Final layout_cost = {final_cost:.6f}")

    side, bbox = bounding_square_side(trees)
    print(f"Bounding square side = {side}")
    print(f"Bounding box (minx, miny, maxx, maxy) = {bbox}")

    out_path = f"greedy_n{args.n}.csv"
    save_layout_to_csv(trees, out_path)
    print(f"Saved layout to {out_path}")


if __name__ == "__main__":
    main()
