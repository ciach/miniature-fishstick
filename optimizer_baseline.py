# optimizer_baseline.py

import math
import random
from copy import deepcopy
from decimal import Decimal

import pandas as pd
from shapely import affinity
from shapely.ops import unary_union
from shapely.strtree import STRtree

from geometry import (
    ChristmasTree,
    layout_cost,
    placed_polygons,
    scale_factor,
)

# Kaggle-like index: 001_0, 002_0, 002_1, ..., 200_0..199
INDEX = [f"{n:03d}_{t}" for n in range(1, 201) for t in range(n)]


def generate_weighted_angle() -> float:
    """
    Same trick as Kaggle:
    random angle with probability proportional to |sin(2*angle)|,
    which biases directions away from axes.
    """
    while True:
        angle = random.uniform(0, 2 * math.pi)
        if random.uniform(0, 1) < abs(math.sin(2 * angle)):
            return angle


def initialize_trees(num_trees: int, existing_trees=None):
    """
    Slightly greedy initializer, adapted from Kaggle's notebook, but using geometry.ChristmasTree.

    Returns:
        placed_trees: list[ChristmasTree]
        side_length: Decimal (unscaled)
    """
    if num_trees == 0:
        return [], Decimal("0")

    if existing_trees is None:
        placed_trees = []
    else:
        # Make a shallow copy so we don't mutate caller state accidentally
        placed_trees = list(existing_trees)

    num_to_add = num_trees - len(placed_trees)
    if num_to_add > 0:
        # create unplaced trees with random angles
        unplaced_trees = [
            ChristmasTree(angle=str(random.uniform(0, 360))) for _ in range(num_to_add)
        ]

        # If starting from scratch, put the first tree at origin
        if not placed_trees:
            placed_trees.append(unplaced_trees.pop(0))

        for tree_to_place in unplaced_trees:
            placed_polys = placed_polygons(placed_trees)
            tree_index = STRtree(placed_polys)

            best_px = None
            best_py = None
            min_radius = Decimal("Infinity")

            for _ in range(10):
                angle = generate_weighted_angle()
                vx = Decimal(str(math.cos(angle)))
                vy = Decimal(str(math.sin(angle)))

                radius = Decimal("20.0")
                step_in = Decimal("0.5")

                collision_found = False
                while radius >= 0:
                    px = radius * vx
                    py = radius * vy

                    candidate_poly = affinity.translate(
                        tree_to_place.polygon,
                        xoff=float(px * scale_factor),
                        yoff=float(py * scale_factor),
                    )

                    possible_indices = tree_index.query(candidate_poly)
                    if any(
                        candidate_poly.intersects(placed_polys[i])
                        and not candidate_poly.touches(placed_polys[i])
                        for i in possible_indices
                    ):
                        collision_found = True
                        break

                    radius -= step_in

                if collision_found:
                    step_out = Decimal("0.05")
                    while True:
                        radius += step_out
                        px = radius * vx
                        py = radius * vy

                        candidate_poly = affinity.translate(
                            tree_to_place.polygon,
                            xoff=float(px * scale_factor),
                            yoff=float(py * scale_factor),
                        )

                        possible_indices = tree_index.query(candidate_poly)
                        if not any(
                            candidate_poly.intersects(placed_polys[i])
                            and not candidate_poly.touches(placed_polys[i])
                            for i in possible_indices
                        ):
                            break
                else:
                    # no collision all the way to center → place at origin
                    radius = Decimal("0")
                    px = Decimal("0")
                    py = Decimal("0")

                if radius < min_radius:
                    min_radius = radius
                    best_px = px
                    best_py = py

            # Commit best position
            tree_to_place.center_x = best_px
            tree_to_place.center_y = best_py
            tree_to_place.polygon = affinity.translate(
                tree_to_place.polygon,
                xoff=float(tree_to_place.center_x * scale_factor),
                yoff=float(tree_to_place.center_y * scale_factor),
            )
            placed_trees.append(tree_to_place)

    # Compute bounding square
    all_polys = placed_polygons(placed_trees)
    bounds = unary_union(all_polys).bounds
    minx = Decimal(str(bounds[0])) / scale_factor
    miny = Decimal(str(bounds[1])) / scale_factor
    maxx = Decimal(str(bounds[2])) / scale_factor
    maxy = Decimal(str(bounds[3])) / scale_factor

    width = maxx - minx
    height = maxy - miny
    side_length = max(width, height)

    return placed_trees, side_length


def build_all_layouts(max_n: int = 200):
    """
    Build layouts for n = 1..max_n using the incremental initializer.
    Returns:
        layouts: dict[int, list[ChristmasTree]]
    """
    layouts = {}
    current_placed = []
    for n in range(1, max_n + 1):
        current_placed, side = initialize_trees(n, existing_trees=current_placed)
        # Store a deepcopy so later modifications don't affect earlier layouts
        layouts[n] = [deepcopy(t) for t in current_placed]
        print(f"Built layout for n={n:3d}, side={side}")
    return layouts


def write_submission(layouts, path: str = "submission.csv"):
    """
    Write Kaggle-style submission CSV from layouts dict.
    layouts: dict[int, list[ChristmasTree]]
    """
    rows = []
    for n in range(1, 201):
        trees = layouts[n]
        for t_idx, tree in enumerate(trees):
            # Kaggle index f"{n:03d}_{t_idx}"
            rows.append(
                {
                    "id": f"{n:03d}_{t_idx}",
                    "x": tree.center_x,
                    "y": tree.center_y,
                    "deg": tree.angle,
                }
            )

    df = pd.DataFrame(rows).set_index("id")
    # Convert to float and round like notebook
    for col in ["x", "y", "deg"]:
        df[col] = df[col].astype(float).round(6)
        df[col] = "s" + df[col].astype("string")

    df.to_csv(path)
    print(f"Saved submission to {path}")


def compute_total_score(layouts) -> float:
    """
    Sum of s^2/n over all puzzles, using layout_cost (which is that,
    plus huge penalties if something breaks).
    """
    total = 0.0
    for n, trees in layouts.items():
        total += layout_cost(trees)
    return total


def main():
    random.seed(42)

    print("Building layouts...")
    layouts = build_all_layouts(max_n=200)

    print("Computing approximate total score...")
    total_score = compute_total_score(layouts)
    print(f"Total score (approx): {total_score:.6f}")

    print("Writing submission CSV...")
    write_submission(layouts, "submission_baseline.csv")


if __name__ == "__main__":
    main()
