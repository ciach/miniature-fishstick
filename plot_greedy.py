#!/usr/bin/env python3

import argparse
import pandas as pd
from decimal import Decimal

from geometry import ChristmasTree
from plot_trees import plot_trees


def load_greedy_csv(path: str):
    df = pd.read_csv(path)
    trees = []

    for _, row in df.iterrows():
        cx = Decimal(str(row["center_x"]))
        cy = Decimal(str(row["center_y"]))
        angle = Decimal(str(row["angle"]))

        t = ChristmasTree(center_x=cx, center_y=cy, angle=angle)
        trees.append(t)

    return trees


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    trees = load_greedy_csv(args.csv)
    plot_trees(
        trees,
        title=f"Plot for {args.csv}",
        show_square=True,
        padding=0.5,
        output_path=args.out,
    )


if __name__ == "__main__":
    main()
