#!/usr/bin/env python3

import math
import random
from copy import deepcopy
from decimal import Decimal
from statistics import median
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from shapely.ops import unary_union
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from geometry import (
    ChristmasTree,
    layout_cost,
    placed_polygons,
    scale_factor,
)

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

MAX_N = 200
RNG_SEED = 42

# SA hyperparams
SA_ITER_SMALL = 1_000  # for small n
SA_ITER_LARGE = 5_000  # for large n
N_THRESHOLD_LARGE = 80

STEP_XY = 0.3
STEP_ANGLE = 10.0

T_START = 1.0
T_END = 1e-3

CENTER_MIN = Decimal("-100")
CENTER_MAX = Decimal("100")

# Multi-start
GOOD_THRESHOLD = 10.0  # if first run is better than this, skip extra restarts
MAX_RESTARTS = 3  # total runs per n (first + up to 2 more)

console = Console()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def random_small_delta(scale: float) -> float:
    """Symmetric small random step in [-scale, +scale]."""
    return (random.random() * 2.0 - 1.0) * scale


def greedy_initialize_single_n(num_trees: int):
    """
    Very simple greedy initializer:

    - Place first tree at origin.
    - For each subsequent tree:
      - Start at radius R on a random angle.
      - Move inwards until collision or center.
      - Back off slightly if collision.

    This is just a starting point for SA, not meant to be amazing.
    """
    if num_trees <= 0:
        return []

    trees = []

    # First tree at origin
    t0 = ChristmasTree(center_x="0", center_y="0", angle=str(random.uniform(0, 360)))
    trees.append(t0)

    for _ in range(1, num_trees):
        new_tree = ChristmasTree(angle=str(random.uniform(0, 360)))

        theta = random.uniform(0, 2 * math.pi)
        vx = math.cos(theta)
        vy = math.sin(theta)

        radius = 20.0
        step_in = 0.5

        placed = placed_polygons(trees)
        collision_found = False

        while radius >= 0:
            cx = radius * vx
            cy = radius * vy

            new_tree.center_x = Decimal(str(cx))
            new_tree.center_y = Decimal(str(cy))
            new_tree.update_polygon()

            candidate = new_tree.polygon
            collided = False
            for poly in placed:
                if candidate.intersects(poly) and not candidate.touches(poly):
                    collided = True
                    break

            if collided:
                collision_found = True
                break

            radius -= step_in

        if collision_found:
            radius += 0.2
            cx = radius * vx
            cy = radius * vy
            new_tree.center_x = Decimal(str(cx))
            new_tree.center_y = Decimal(str(cy))
            new_tree.update_polygon()
        else:
            new_tree.center_x = Decimal("0")
            new_tree.center_y = Decimal("0")
            new_tree.update_polygon()

        trees.append(deepcopy(new_tree))

    return trees


def compact_layout(trees):
    """
    Simple compaction: shift all trees so the bounding box moves near (0, 0).

    This does NOT shrink the square like a true compactor, but avoids them
    drifting too far away.
    """
    if not trees:
        return

    polys = placed_polygons(trees)
    bounds = unary_union(polys).bounds  # scaled
    minx = Decimal(str(bounds[0])) / scale_factor
    miny = Decimal(str(bounds[1])) / scale_factor

    dx = -minx
    dy = -miny

    for t in trees:
        t.center_x += dx
        t.center_y += dy
        t.update_polygon()


def simulated_annealing_for_n(n: int):
    """
    Run SA for a specific puzzle size n.

    Returns:
        best_trees: list[ChristmasTree]
        best_cost: float
    """
    # Initial layout
    current = greedy_initialize_single_n(n)
    compact_layout(current)

    current_cost = layout_cost(current)
    best = deepcopy(current)
    best_cost = current_cost

    max_iter = SA_ITER_SMALL if n < N_THRESHOLD_LARGE else SA_ITER_LARGE

    for step in range(max_iter):
        # Cooling schedule
        alpha = step / max(1, max_iter - 1)  # 0 -> 1
        T = T_START * (T_END / T_START) ** alpha

        # Adaptive step sizes: big moves early, small moves late
        scale = max(0.1, 1.0 - alpha)
        dx = random_small_delta(STEP_XY * scale)
        dy = random_small_delta(STEP_XY * scale)
        dtheta = random_small_delta(STEP_ANGLE * scale)

        idx = random.randrange(len(current))
        tree = current[idx]

        # Save old state
        old_cx = tree.center_x
        old_cy = tree.center_y
        old_angle = tree.angle

        # Propose move
        tree.center_x = tree.center_x + Decimal(str(dx))
        tree.center_y = tree.center_y + Decimal(str(dy))
        tree.angle = tree.angle + Decimal(str(dtheta))
        tree.update_polygon()

        # Hard constraints: centers must stay in [-100, 100]
        if (
            tree.center_x < CENTER_MIN
            or tree.center_x > CENTER_MAX
            or tree.center_y < CENTER_MIN
            or tree.center_y > CENTER_MAX
        ):
            # reject immediately
            tree.center_x = old_cx
            tree.center_y = old_cy
            tree.angle = old_angle
            tree.update_polygon()
            continue

        # Occasionally re-anchor layout (cheap compaction)
        if step % 200 == 0:
            compact_layout(current)

        new_cost = layout_cost(current)
        delta = new_cost - current_cost

        accept = False
        if delta <= 0:
            accept = True
        else:
            if T > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    accept = True

        if accept:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best = deepcopy(current)
        else:
            # revert to old state
            tree.center_x = old_cx
            tree.center_y = old_cy
            tree.angle = old_angle
            tree.update_polygon()

    return best, best_cost


# ----------------------------------------------------------------------
# Multiprocessing worker & CSV writer
# ----------------------------------------------------------------------


def sa_worker(n: int):
    """
    Worker for ProcessPoolExecutor.

    Runs SA for this n with multi-start:
      - Always at least 1 run.
      - If first run's cost > GOOD_THRESHOLD, do up to MAX_RESTARTS-1 extra runs.

    Returns:
        n,
        layout: list of (center_x_str, center_y_str, angle_str),
        best_cost
    """
    best_cost = float("inf")
    best_trees: list[ChristmasTree] | None = None

    for restart in range(MAX_RESTARTS):
        # Per-(n, restart) seed for reproducibility
        random.seed(RNG_SEED + n * 1000 + restart)

        trees, cost = simulated_annealing_for_n(n)

        if cost < best_cost:
            best_cost = cost
            best_trees = trees

        # If first run already "good enough", don't waste more time
        if restart == 0 and best_cost <= GOOD_THRESHOLD:
            break

    assert best_trees is not None
    layout = [(str(t.center_x), str(t.center_y), str(t.angle)) for t in best_trees]
    return n, layout, best_cost


def write_submission(layouts, path: str = "submission_sa.csv"):
    """
    Write Kaggle-style submission CSV from compressed layouts.

    layouts: dict[int, list[(cx_str, cy_str, angle_str)]]
    """
    rows = []
    for n in range(1, MAX_N + 1):
        layout = layouts[n]
        for t_idx, (cx_str, cy_str, angle_str) in enumerate(layout):
            rows.append(
                {
                    "id": f"{n:03d}_{t_idx}",
                    "x": cx_str,
                    "y": cy_str,
                    "deg": angle_str,
                }
            )

    df = pd.DataFrame(rows).set_index("id")

    for col in ["x", "y", "deg"]:
        df[col] = df[col].astype(float).round(6)
        df[col] = "s" + df[col].astype("string")

    df.to_csv(path)
    console.print(f"[green]Saved SA submission to [bold]{path}[/bold][/green]")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    ns = list(range(1, MAX_N + 1))

    layouts: dict[int, list[tuple[str, str, str]]] = {}
    best_costs: dict[int, float] = {}
    total_cost = 0.0

    console.print(
        "[bold cyan]Starting SA optimization with multiprocessing...[/bold cyan]"
    )

    executor = ProcessPoolExecutor()
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total} n"),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Optimizing", total=len(ns))

            futures = {executor.submit(sa_worker, n): n for n in ns}

            for fut in as_completed(futures):
                n = futures[fut]
                try:
                    n_ret, layout, best_cost = fut.result()
                except Exception as e:
                    console.print(f"[red]Worker for n={n} failed: {e}[/red]")
                    progress.update(task, advance=1)
                    continue

                layouts[n_ret] = layout
                best_costs[n_ret] = best_cost
                total_cost += best_cost

                progress.update(
                    task,
                    advance=1,
                    description=f"Optimizing (last finished n={n_ret})",
                )
                console.log(f"Finished n={n_ret:3d}, best_cost={best_cost:.6f}")

    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Ctrl+C detected. Cancelling all workers and shutting down...[/yellow]"
        )
        executor.shutdown(wait=False, cancel_futures=True)
        raise SystemExit(1)
    else:
        executor.shutdown(wait=True)

        # Summary stats
        if best_costs:
            ordered = [best_costs[n] for n in sorted(best_costs)]
            min_cost = min(ordered)
            max_cost = max(ordered)
            med_cost = median(ordered)

            console.print(
                f"\n[bold green]Approx total SA score (sum s²/n over n=1..{MAX_N}): "
                f"{total_cost:.6f}[/bold green]"
            )
            console.print(
                f"[cyan]Cost summary:[/cyan] "
                f"min={min_cost:.6f}, median={med_cost:.6f}, max={max_cost:.6f}"
            )

            # Show a few worst offenders
            worst = sorted(best_costs.items(), key=lambda kv: kv[1], reverse=True)[:10]
            console.print("[magenta]Worst n by best_cost:[/magenta]")
            for n, c in worst:
                console.print(f"  n={n:3d}, best_cost={c:.6f}")
        else:
            console.print("[red]No results collected. Something went wrong.[/red]")

        write_submission(layouts, "submission_sa.csv")


if __name__ == "__main__":
    main()
