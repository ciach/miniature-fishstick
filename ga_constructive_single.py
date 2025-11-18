#!/usr/bin/env python3

import argparse
import random
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from statistics import median
from typing import List, Tuple, Optional, Dict

import pandas as pd

from geometry import (
    ChristmasTree,
    layout_cost,
)

# ------------------------------------------------------------
# GA / heuristic config
# ------------------------------------------------------------

NUM_ANGLE_BUCKETS = 16  # 360 / 16 = 22.5° resolution
GRID_RADIUS = 5.0  # search square [-R, R] in x, y
GRID_STEP = 0.6  # coarser grid → fewer candidates

POP_SIZE = 20  # smaller pop for prototype
NUM_GENERATIONS = 30  # fewer generations to see progress
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.2

INFEASIBLE_PENALTY = 1e6  # big but not insane


@dataclass
class Individual:
    order: List[int]  # permutation of 0..n-1
    angles: List[int]  # bucket index 0..NUM_ANGLE_BUCKETS-1
    fitness: Optional[float] = None


# ------------------------------------------------------------
# Geometry helpers for constructive packing
# ------------------------------------------------------------


def frange(start: float, stop: float, step: float):
    """Simple float range generator."""
    x = start
    eps = step * 0.5
    while x <= stop + eps:
        yield x
        x += step


def generate_candidate_positions(
    radius: float, step: float
) -> List[Tuple[float, float]]:
    """
    Generate a coarse grid of candidate positions in [-radius, radius]^2,
    sorted by distance from origin so we try more central positions first.
    """
    pts: List[Tuple[float, float]] = []
    for x in frange(-radius, radius, step):
        for y in frange(-radius, radius, step):
            pts.append((float(x), float(y)))
    pts.sort(key=lambda p: p[0] * p[0] + p[1] * p[1])
    return pts


def trees_overlap(tree: ChristmasTree, others: List[ChristmasTree]) -> bool:
    """
    Detect any true overlaps (intersects but not just touches).
    """
    candidate = tree.polygon
    for t in others:
        poly = t.polygon
        if candidate.intersects(poly) and not candidate.touches(poly):
            return True
    return False


def angle_from_bucket(bucket: int) -> float:
    """Convert angle bucket index to degrees."""
    bucket = bucket % NUM_ANGLE_BUCKETS
    return (360.0 / NUM_ANGLE_BUCKETS) * bucket


def construct_layout_for_individual(
    n: int,
    ind: Individual,
    candidate_positions: List[Tuple[float, float]],
) -> Tuple[Optional[List[ChristmasTree]], float]:
    """
    Construct a layout for this individual's (order, angles) using a greedy
    placement on a fixed grid.

    IMPORTANT: to keep it fast, we:
      - place each tree at the FIRST feasible candidate
      - only call layout_cost ONCE at the end
    """
    placed: List[ChristmasTree] = []

    for idx in ind.order:
        bucket = ind.angles[idx]
        angle_deg = angle_from_bucket(bucket)

        placed_tree: Optional[ChristmasTree] = None

        # Try candidate positions in increasing distance from origin
        for x, y in candidate_positions:
            t = ChristmasTree(
                center_x=Decimal(str(x)),
                center_y=Decimal(str(y)),
                angle=Decimal(str(angle_deg)),
            )

            if trees_overlap(t, placed):
                continue

            # First non-overlapping position: accept it
            placed_tree = t
            break

        if placed_tree is None:
            # No feasible position on our grid → infeasible layout
            return None, INFEASIBLE_PENALTY

        placed.append(placed_tree)

    # All placed, compute full cost once
    cost = layout_cost(placed)
    return placed, cost


# ------------------------------------------------------------
# GA components
# ------------------------------------------------------------


def init_individual(n: int) -> Individual:
    order = list(range(n))
    random.shuffle(order)
    angles = [random.randrange(NUM_ANGLE_BUCKETS) for _ in range(n)]
    return Individual(order=order, angles=angles, fitness=None)


def evaluate_individual(
    n: int,
    ind: Individual,
    candidate_positions: List[Tuple[float, float]],
    cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float],
) -> float:
    """
    Evaluate fitness of an individual with caching to avoid re-evaluations.
    """
    key = (tuple(ind.order), tuple(ind.angles))
    if key in cache:
        ind.fitness = cache[key]
        return ind.fitness

    _, cost = construct_layout_for_individual(n, ind, candidate_positions)
    ind.fitness = cost
    cache[key] = cost
    return cost


def tournament_select(pop: List[Individual], k: int) -> Individual:
    """k-way tournament selection (minimization)."""
    chosen = random.sample(pop, k)
    best = min(
        chosen, key=lambda ind: ind.fitness if ind.fitness is not None else float("inf")
    )
    return deepcopy(best)


def crossover_order(p1_order: List[int], p2_order: List[int]) -> List[int]:
    """
    Order crossover (OX) for permutation.
    """
    n = len(p1_order)
    c1, c2 = sorted(random.sample(range(n), 2))
    child = [None] * n  # type: ignore

    # Copy slice from parent1
    child[c1:c2] = p1_order[c1:c2]

    # Fill remaining positions from parent2 in order
    p2_idx = 0
    for i in range(n):
        if child[i] is None:
            while p2_order[p2_idx] in child:
                p2_idx += 1
            child[i] = p2_order[p2_idx]
            p2_idx += 1

    return child  # type: ignore


def crossover_angles(p1_angles: List[int], p2_angles: List[int]) -> List[int]:
    """
    Uniform crossover for angle buckets.
    """
    n = len(p1_angles)
    child = []
    for i in range(n):
        if random.random() < 0.5:
            child.append(p1_angles[i])
        else:
            child.append(p2_angles[i])
    return child


def crossover(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
    """
    Crossover of two individuals → two children.
    """
    child1_order = crossover_order(p1.order, p2.order)
    child2_order = crossover_order(p2.order, p1.order)

    child1_angles = crossover_angles(p1.angles, p2.angles)
    child2_angles = crossover_angles(p2.angles, p1.angles)

    return (
        Individual(order=child1_order, angles=child1_angles, fitness=None),
        Individual(order=child2_order, angles=child2_angles, fitness=None),
    )


def mutate(
    ind: Individual, n: int, p_swap: float = 0.1, p_angle_mut: float = 0.1
) -> Individual:
    """
    Mutate order (swap) and angles (re-randomize some buckets).
    """
    # Mutate order: swap two positions with probability p_swap
    if random.random() < p_swap:
        i, j = random.sample(range(n), 2)
        ind.order[i], ind.order[j] = ind.order[j], ind.order[i]

    # Mutate angles: each gene has p_angle_mut chance to change
    for i in range(n):
        if random.random() < p_angle_mut:
            ind.angles[i] = random.randrange(NUM_ANGLE_BUCKETS)

    ind.fitness = None
    return ind


def run_ga_for_n(
    n: int,
    population_size: int = POP_SIZE,
    generations: int = NUM_GENERATIONS,
) -> Tuple[Individual, List[float]]:
    """
    Run GA for a single puzzle size n.

    Returns:
        best_individual, history_of_best_costs
    """
    candidate_positions = generate_candidate_positions(GRID_RADIUS, GRID_STEP)
    cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}

    print(f"Generating {len(candidate_positions)} candidate positions...")

    # Initialize population
    pop: List[Individual] = [init_individual(n) for _ in range(population_size)]
    print(f"Evaluating initial population of size {population_size} for n={n}...")
    for idx, ind in enumerate(pop, start=1):
        evaluate_individual(n, ind, candidate_positions, cache)
        if idx % 5 == 0 or idx == population_size:
            print(f"  initial eval {idx}/{population_size}")

    best = min(
        pop, key=lambda ind: ind.fitness if ind.fitness is not None else float("inf")
    )
    best = deepcopy(best)
    best_history: List[float] = [
        best.fitness if best.fitness is not None else float("inf")
    ]

    print(f"Initial best cost for n={n}: {best_history[-1]:.6f}")

    # Main GA loop
    for gen in range(1, generations + 1):
        new_pop: List[Individual] = []

        while len(new_pop) < population_size:
            p1 = tournament_select(pop, TOURNAMENT_K)
            p2 = tournament_select(pop, TOURNAMENT_K)

            if random.random() < CROSSOVER_RATE:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = deepcopy(p1), deepcopy(p2)

            if random.random() < MUTATION_RATE:
                c1 = mutate(c1, n)
            if random.random() < MUTATION_RATE:
                c2 = mutate(c2, n)

            new_pop.append(c1)
            if len(new_pop) < population_size:
                new_pop.append(c2)

        pop = new_pop

        # Evaluate new population
        for ind in pop:
            if ind.fitness is None:
                evaluate_individual(n, ind, candidate_positions, cache)

        gen_best = min(
            pop,
            key=lambda ind: ind.fitness if ind.fitness is not None else float("inf"),
        )
        if gen_best.fitness is not None and gen_best.fitness < (
            best.fitness or float("inf")
        ):
            best = deepcopy(gen_best)

        best_history.append(best.fitness if best.fitness is not None else float("inf"))
        print(f"Gen {gen:03d}, best_cost={best_history[-1]:.6f}")

    return best, best_history


# ------------------------------------------------------------
# Export / CLI
# ------------------------------------------------------------


def build_layout_from_individual(n: int, ind: Individual) -> List[ChristmasTree]:
    """
    Build final layout (without penalty) for the best individual.
    """
    candidate_positions = generate_candidate_positions(GRID_RADIUS, GRID_STEP)
    trees, cost = construct_layout_for_individual(n, ind, candidate_positions)
    if trees is None:
        raise RuntimeError(f"Best individual for n={n} is infeasible (cost={cost})")
    return trees


def write_single_n_csv(n: int, trees: List[ChristmasTree], path: str) -> None:
    """
    Write a Kaggle-style CSV for a single n.
    """
    rows = []
    for t_idx, t in enumerate(trees):
        rows.append(
            {
                "id": f"{n:03d}_{t_idx}",
                "x": float(t.center_x),
                "y": float(t.center_y),
                "deg": float(t.angle),
            }
        )

    df = pd.DataFrame(rows).set_index("id")
    for col in ["x", "y", "deg"]:
        df[col] = df[col].astype(float).round(6)
        df[col] = "s" + df[col].astype("string")

    df.to_csv(path)
    print(f"Saved GA layout for n={n} to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="GA + constructive heuristic for a single n"
    )
    parser.add_argument(
        "--n", type=int, default=80, help="Puzzle size n (e.g. 80 or 120)"
    )
    parser.add_argument("--pop", type=int, default=POP_SIZE, help="Population size")
    parser.add_argument(
        "--gens", type=int, default=NUM_GENERATIONS, help="Number of generations"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to save GA layout CSV for this n.",
    )
    args = parser.parse_args()

    random.seed(1234 + args.n)

    best, history = run_ga_for_n(
        args.n, population_size=args.pop, generations=args.gens
    )
    print(f"\nBest fitness for n={args.n}: {best.fitness:.6f}")

    trees = build_layout_from_individual(args.n, best)
    print(f"layout_cost(best_layout) = {layout_cost(trees):.6f}")

    if args.csv:
        write_single_n_csv(args.n, trees, args.csv)


if __name__ == "__main__":
    main()
