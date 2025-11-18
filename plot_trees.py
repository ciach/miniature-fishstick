#!/usr/bin/env python3

from decimal import Decimal, getcontext
from typing import List

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.ops import unary_union

from geometry import ChristmasTree, placed_polygons, scale_factor

# Keep precision consistent with geometry.py
getcontext().prec = 25


def plot_trees(
    trees: List[ChristmasTree],
    title: str | None = None,
    show_square: bool = True,
    padding: float = 0.5,
    output_path: str | None = None,
) -> None:
    """
    Visualize current tree layout in unscaled coordinates, similar to Kaggle's plot.

    - Draws each tree polygon.
    - Optionally draws the minimal bounding square around all trees.
    - If output_path is given, saves the plot to that file.
    """
    if not trees:
        print("No trees to plot.")
        return

    polys = placed_polygons(trees)
    bounds = unary_union(polys).bounds  # scaled floats: (minx, miny, maxx, maxy)

    # Convert bounds to unscaled Decimals
    minx = Decimal(str(bounds[0])) / scale_factor
    miny = Decimal(str(bounds[1])) / scale_factor
    maxx = Decimal(str(bounds[2])) / scale_factor
    maxy = Decimal(str(bounds[3])) / scale_factor

    width = maxx - minx
    height = maxy - miny
    side_length = max(width, height)

    fig, ax = plt.subplots(figsize=(6, 6))

    num_trees = len(trees)
    if num_trees > 1:
        colors = [plt.cm.viridis(i / (num_trees - 1)) for i in range(num_trees)]
    else:
        colors = [plt.cm.viridis(0.5)]

    # Plot each tree
    for i, tree in enumerate(trees):
        x_scaled, y_scaled = tree.polygon.exterior.xy  # scaled floats
        x_unscaled = [Decimal(str(v)) / scale_factor for v in x_scaled]
        y_unscaled = [Decimal(str(v)) / scale_factor for v in y_scaled]

        xs = [float(v) for v in x_unscaled]
        ys = [float(v) for v in y_unscaled]

        ax.plot(xs, ys, color=colors[i])
        ax.fill(xs, ys, alpha=0.5, color=colors[i])

    # Draw bounding square if requested
    if show_square:
        # Center the square on the larger dimension
        if width >= height:
            square_x = minx
            square_y = miny - (side_length - height) / Decimal("2")
        else:
            square_x = minx - (side_length - width) / Decimal("2")
            square_y = miny

        bounding_square = Rectangle(
            (float(square_x), float(square_y)),
            float(side_length),
            float(side_length),
            fill=False,
            edgecolor="red",
            linewidth=2,
            linestyle="--",
        )
        ax.add_patch(bounding_square)

        x_min_plot = square_x - Decimal(str(padding))
        x_max_plot = square_x + side_length + Decimal(str(padding))
        y_min_plot = square_y - Decimal(str(padding))
        y_max_plot = square_y + side_length + Decimal(str(padding))
    else:
        x_min_plot = minx - Decimal(str(padding))
        x_max_plot = maxx + Decimal(str(padding))
        y_min_plot = miny - Decimal(str(padding))
        y_max_plot = maxy + Decimal(str(padding))

    ax.set_xlim(float(x_min_plot), float(x_max_plot))
    ax.set_ylim(float(y_min_plot), float(y_max_plot))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    if title is None:
        side_str = f"{side_length:.12f}"
        ax.set_title(f"{num_trees} trees | side = {side_str}")
    else:
        ax.set_title(title)

    # Save or show
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Saved plot to {output_path}")
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"Could not show figure: {e}")

    plt.close(fig)


def demo_layout(num_trees: int = 5) -> List[ChristmasTree]:
    """
    Simple demo layout: place N trees roughly on a circle so we can see something.
    This is just for quick testing, not for Kaggle logic.
    """
    import math

    trees: List[ChristmasTree] = []
    radius = Decimal("2.0")

    for i in range(num_trees):
        angle_deg = Decimal(str(360 * i / max(1, num_trees)))
        theta = math.radians(float(angle_deg))
        cx = radius * Decimal(str(math.cos(theta)))
        cy = radius * Decimal(str(math.sin(theta)))

        t = ChristmasTree(center_x=cx, center_y=cy, angle=angle_deg)
        trees.append(t)

    return trees


if __name__ == "__main__":
    # Quick local test
    trees = demo_layout(num_trees=6)
    plot_trees(trees, title="Demo layout: 6 trees", output_path="demo_layout.png")
