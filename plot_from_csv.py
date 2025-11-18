#!/usr/bin/env python3

import argparse
from decimal import Decimal

import pandas as pd

from geometry import ChristmasTree
from plot_trees import plot_trees


def load_layout_from_csv(csv_path: str, n: int):
    """
    Load layout for puzzle with n trees from a Kaggle-style submission CSV.

    id format: 'NNN_k', so we filter rows starting with f'{n:03d}_'.
    """
    df = pd.read_csv(csv_path, index_col="id")

    prefix = f"{n:03d}_"
    df_n = df[df.index.str.startswith(prefix)]

    trees = []

    for idx, row in df_n.iterrows():
        # values look like 's0.123456'
        def parse_s(val: str) -> Decimal:
            val = str(val)
            if val.startswith("s"):
                val = val[1:]
            return Decimal(val)

        cx = parse_s(row["x"])
        cy = parse_s(row["y"])
        angle = parse_s(row["deg"])

        tree = ChristmasTree(center_x=cx, center_y=cy, angle=angle)
        trees.append(tree)

    return trees


def main():
    parser = argparse.ArgumentParser(
        description="Plot Santa packing layout from a submission CSV."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="submission_baseline.csv",
        help="Path to Kaggle-style submission CSV.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Puzzle size to plot (1..200).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to save the plot image (e.g. trees_200.png). "
        "If not set, will try to show the figure.",
    )

    args = parser.parse_args()

    trees = load_layout_from_csv(args.csv, args.n)
    if not trees:
        print(f"No rows found for n={args.n} in {args.csv}")
        return

    title = f"{args.n} trees from {args.csv}"
    plot_trees(
        trees,
        title=title,
        show_square=True,
        padding=0.5,
        output_path=args.out,
    )


if __name__ == "__main__":
    main()
