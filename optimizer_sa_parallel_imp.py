#!/usr/bin/env python3

import json
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import median

import pandas as pd
from shapely.ops import unary_union
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from geometry import (
    ChristmasTree,
    has_overlap,
    layout_cost,
    placed_polygons,
    scale_factor,
)

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

MAX_N = 200
RNG_SEED = 42
N_THRESHOLD_LARGE = 80

STEP_XY = 0.3
STEP_ANGLE = 10.0
CLUSTER_ANGLE = 5.0
CLUSTER_SIZE = 3

T_START = 1.0
T_END = 1e-3
ACCEPT_WINDOW = 50
ACCEPT_RATE_LOW = 0.2
ACCEPT_RATE_HIGH = 0.65
FAST_COOL_FACTOR = 0.85
REHEAT_FACTOR = 1.15

CENTER_MIN = Decimal("-100")
CENTER_MAX = Decimal("100")

HEAVY_RERUN_THRESHOLD = 10.0
RESULTS_PATH = Path("sa_partial_results.json")
GUIDED_MOVE_PROB = 0.25
GUIDED_STEP = 0.6
USE_PATTERN_INIT = True
PATTERN_SPACING = 1.4
USE_HEX_INIT = True
HEX_SPACING_X = 1.35
HEX_SPACING_Y = 1.15
USE_REPULSION_INIT = True
REPULSION_ITERS = 200
REPULSION_STEP = 0.35
KICK_THRESHOLD = 250
KICK_MAGNITUDE = 1.0
KICK_FRACTION = 0.15
MOVE_WEIGHTS = {
    "single": 0.6,
    "swap": 0.2,
    "cluster": 0.2,
}
SCALE = float(scale_factor)


@dataclass(frozen=True)
class SASettings:
    label: str
    iter_small: int
    iter_large: int
    max_restarts: int
    good_threshold: float
    seed_offset: int = 0


LIGHT_SETTINGS = SASettings(
    label="light",
    iter_small=200,
    iter_large=600,
    max_restarts=1,
    good_threshold=10.0,
    seed_offset=0,
)

HEAVY_SETTINGS = SASettings(
    label="heavy",
    iter_small=500,
    iter_large=2000,
    max_restarts=2,
    good_threshold=5.0,
    seed_offset=1_000_000,
)

console = Console()


def adjust_temperature(current_temp: float, acceptance_rate: float) -> float:
    """Adaptively reheat/cool based on observed acceptance rate."""
    updated = current_temp
    if acceptance_rate > ACCEPT_RATE_HIGH:
        updated *= FAST_COOL_FACTOR
    elif acceptance_rate < ACCEPT_RATE_LOW:
        updated *= REHEAT_FACTOR

    # Clamp so we do not overshoot extremes too aggressively
    if updated < T_END:
        return T_END
    if updated > T_START:
        return T_START
    return updated


def load_partial_results() -> dict[int, dict]:
    if not RESULTS_PATH.exists():
        return {}

    data = json.loads(RESULTS_PATH.read_text())
    results: dict[int, dict] = {}
    for key, entry in data.items():
        try:
            n = int(key)
        except ValueError:
            continue

        layout_raw = entry.get("layout", [])
        layout = [tuple(item) for item in layout_raw]
        best_cost = float(entry.get("best_cost", float("inf")))
        best_settings = entry.get("best_settings", entry.get("settings", "unknown"))
        heavy_attempted = bool(
            entry.get(
                "heavy_attempted",
                best_settings == HEAVY_SETTINGS.label,
            )
        )

        results[n] = {
            "layout": layout,
            "best_cost": best_cost,
            "best_settings": best_settings,
            "heavy_attempted": heavy_attempted,
        }

    return results


def persist_partial_results(results: dict[int, dict]) -> None:
    serializable: dict[str, dict] = {}
    for n, entry in sorted(results.items()):
        layout_serializable = [list(triple) for triple in entry["layout"]]
        serializable[str(n)] = {
            "layout": layout_serializable,
            "best_cost": entry["best_cost"],
            "best_settings": entry.get("best_settings", "unknown"),
            "heavy_attempted": entry.get("heavy_attempted", False),
        }

    RESULTS_PATH.write_text(json.dumps(serializable, indent=2))


def record_result(
    n: int,
    layout: list[tuple[str, str, str]],
    best_cost: float,
    settings: SASettings,
    results: dict[int, dict],
) -> None:
    existing = results.get(n)
    improved = existing is None or best_cost < existing["best_cost"]

    heavy_attempted = (
        existing.get("heavy_attempted", False) if existing else False
    ) or (settings.label == HEAVY_SETTINGS.label)

    if improved or existing is None:
        new_entry = {
            "layout": [tuple(triple) for triple in layout],
            "best_cost": best_cost,
            "best_settings": settings.label,
            "heavy_attempted": heavy_attempted,
        }
    else:
        new_entry = {
            "layout": existing["layout"],
            "best_cost": existing["best_cost"],
            "best_settings": existing.get("best_settings", settings.label),
            "heavy_attempted": heavy_attempted,
        }

    results[n] = new_entry
    persist_partial_results(results)


def needs_light_run(n: int, results: dict[int, dict]) -> bool:
    return n not in results


def needs_heavy_run(n: int, results: dict[int, dict]) -> bool:
    entry = results.get(n)
    if entry is None:
        return True
    return entry["best_cost"] > HEAVY_RERUN_THRESHOLD and not entry.get(
        "heavy_attempted", False
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def random_small_delta(scale: float) -> float:
    """Symmetric small random step in [-scale, +scale]."""
    return (random.random() * 2.0 - 1.0) * scale


def capture_state(tree: ChristmasTree) -> tuple[Decimal, Decimal, Decimal]:
    return (tree.center_x, tree.center_y, tree.angle)


def restore_states(trees: list[ChristmasTree], states) -> None:
    for idx, state in states:
        cx, cy, angle = state
        tree = trees[idx]
        tree.center_x = cx
        tree.center_y = cy
        tree.angle = angle
        tree.update_polygon()


def compute_guided_move(
    trees: list[ChristmasTree],
    idx: int,
    step: float = GUIDED_STEP,
) -> tuple[float, float] | None:
    """Return a push vector pointing away from the overlap centroid."""
    target = trees[idx]
    target_poly = target.polygon
    target_cx = float(target.center_x)
    target_cy = float(target.center_y)

    for other_idx, other in enumerate(trees):
        if other_idx == idx:
            continue

        poly = other.polygon
        if not (target_poly.intersects(poly) and not target_poly.touches(poly)):
            continue

        inter = target_poly.intersection(poly)
        centroid = inter.centroid
        centroid_x = float(centroid.x) / SCALE
        centroid_y = float(centroid.y) / SCALE

        vx = target_cx - centroid_x
        vy = target_cy - centroid_y

        norm = math.hypot(vx, vy)
        if norm < 1e-9:
            vx = target_cx - float(other.center_x)
            vy = target_cy - float(other.center_y)
            norm = math.hypot(vx, vy)

        if norm < 1e-9:
            angle = random.random() * 2.0 * math.pi
            vx = math.cos(angle)
            vy = math.sin(angle)
            norm = 1.0

        return (vx / norm) * step, (vy / norm) * step

    return None


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


def hex_grid_initialize_single_n(num_trees: int) -> list[ChristmasTree]:
    if num_trees <= 0:
        return []

    cols = int(math.ceil(math.sqrt(num_trees)))
    rows = int(math.ceil(num_trees / cols))
    trees: list[ChristmasTree] = []
    idx = 0
    for r in range(rows):
        row_offset = (HEX_SPACING_X * 0.5) if (r % 2 == 1) else 0.0
        for c in range(cols):
            if idx >= num_trees:
                break
            x = (c - cols / 2) * HEX_SPACING_X + row_offset
            y = (r - rows / 2) * HEX_SPACING_Y
            angle = (idx * 13.0) % 360.0
            tree = ChristmasTree(center_x=str(x), center_y=str(y), angle=str(angle))
            trees.append(tree)
            idx += 1

    compact_layout(trees)
    if has_overlap(trees):
        return []
    return trees


def repulsion_initialize_single_n(num_trees: int) -> list[ChristmasTree]:
    if num_trees <= 0:
        return []

    trees = []
    for i in range(num_trees):
        angle = (i * 31.0) % 360.0
        trees.append(ChristmasTree(center_x="0", center_y="0", angle=str(angle)))

    for _ in range(REPULSION_ITERS):
        overlaps = 0
        for i in range(num_trees):
            for j in range(i + 1, num_trees):
                t1 = trees[i]
                t2 = trees[j]
                if t1.polygon.intersects(t2.polygon) and not t1.polygon.touches(t2.polygon):
                    overlaps += 1
                    dx = float(t1.center_x - t2.center_x)
                    dy = float(t1.center_y - t2.center_y)
                    dist = math.hypot(dx, dy)
                    if dist < 1e-6:
                        angle = random.random() * 2.0 * math.pi
                        dx = math.cos(angle)
                        dy = math.sin(angle)
                        dist = 1.0
                    step = REPULSION_STEP / dist
                    shift_x = dx * step
                    shift_y = dy * step
                    t1.center_x += Decimal(str(shift_x))
                    t1.center_y += Decimal(str(shift_y))
                    t2.center_x -= Decimal(str(shift_x))
                    t2.center_y -= Decimal(str(shift_y))
                    t1.update_polygon()
                    t2.update_polygon()
        if overlaps == 0:
            break

    compact_layout(trees)
    if has_overlap(trees):
        return []
    return trees


def pattern_initialize_single_n(num_trees: int, spacing: float = PATTERN_SPACING):
    """Place trees along a Fermat spiral to get a compact seed layout."""
    if num_trees <= 0:
        return []

    trees = []
    golden_angle = math.pi * (3.0 - math.sqrt(5))

    for idx in range(num_trees):
        radius = spacing * math.sqrt(idx)
        theta = idx * golden_angle
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        angle = (idx * 17.0) % 360.0
        tree = ChristmasTree(center_x=str(x), center_y=str(y), angle=str(angle))
        trees.append(tree)

    compact_layout(trees)
    if has_overlap(trees):
        return greedy_initialize_single_n(num_trees)
    return trees


def choose_initial_layout(num_trees: int) -> list[ChristmasTree]:
    initializers = []
    if USE_HEX_INIT:
        initializers.append(hex_grid_initialize_single_n)
    if USE_REPULSION_INIT:
        initializers.append(repulsion_initialize_single_n)
    if USE_PATTERN_INIT:
        initializers.append(pattern_initialize_single_n)

    for init_fn in initializers:
        layout = init_fn(num_trees)
        if layout:
            return layout

    return greedy_initialize_single_n(num_trees)


def kick_layout(trees: list[ChristmasTree]) -> None:
    if not trees:
        return

    count = max(1, int(len(trees) * KICK_FRACTION))
    indices = random.sample(range(len(trees)), count)

    for idx in indices:
        tree = trees[idx]
        tree.center_x += Decimal(str(random_small_delta(KICK_MAGNITUDE)))
        tree.center_y += Decimal(str(random_small_delta(KICK_MAGNITUDE)))
        tree.angle += Decimal(str(random_small_delta(25.0)))
        tree.center_x = max(CENTER_MIN, min(CENTER_MAX, tree.center_x))
        tree.center_y = max(CENTER_MIN, min(CENTER_MAX, tree.center_y))
        tree.update_polygon()

    compact_layout(trees)


def apply_single_move(
    trees: list[ChristmasTree], idx: int, dx: float, dy: float, dtheta: float
) -> list[tuple[int, tuple[Decimal, Decimal, Decimal]]] | None:
    tree = trees[idx]
    state = capture_state(tree)
    tree.center_x = tree.center_x + Decimal(str(dx))
    tree.center_y = tree.center_y + Decimal(str(dy))
    tree.angle = tree.angle + Decimal(str(dtheta))

    if (
        tree.center_x < CENTER_MIN
        or tree.center_x > CENTER_MAX
        or tree.center_y < CENTER_MIN
        or tree.center_y > CENTER_MAX
    ):
        tree.center_x, tree.center_y, tree.angle = state
        tree.update_polygon()
        return None

    tree.update_polygon()
    return [(idx, state)]


def apply_swap_move(
    trees: list[ChristmasTree],
) -> list[tuple[int, tuple[Decimal, Decimal, Decimal]]] | None:
    if len(trees) < 2:
        return None

    i, j = random.sample(range(len(trees)), 2)
    state_i = capture_state(trees[i])
    state_j = capture_state(trees[j])

    trees[i].center_x, trees[j].center_x = trees[j].center_x, trees[i].center_x
    trees[i].center_y, trees[j].center_y = trees[j].center_y, trees[i].center_y
    trees[i].angle, trees[j].angle = trees[j].angle, trees[i].angle

    trees[i].update_polygon()
    trees[j].update_polygon()
    return [(i, state_i), (j, state_j)]


def apply_cluster_rotate_move(
    trees: list[ChristmasTree],
    size: int = CLUSTER_SIZE,
    angle_scale: float = CLUSTER_ANGLE,
) -> list[tuple[int, tuple[Decimal, Decimal, Decimal]]] | None:
    if not trees:
        return None

    count = min(size, len(trees))
    indices = random.sample(range(len(trees)), count)
    centroid_x = sum(float(trees[i].center_x) for i in indices) / count
    centroid_y = sum(float(trees[i].center_y) for i in indices) / count
    dtheta = math.radians(random_small_delta(angle_scale))

    states = []
    for idx in indices:
        tree = trees[idx]
        state = capture_state(tree)
        rel_x = float(tree.center_x) - centroid_x
        rel_y = float(tree.center_y) - centroid_y
        new_x = rel_x * math.cos(dtheta) - rel_y * math.sin(dtheta) + centroid_x
        new_y = rel_x * math.sin(dtheta) + rel_y * math.cos(dtheta) + centroid_y
        tree.center_x = Decimal(str(new_x))
        tree.center_y = Decimal(str(new_y))
        tree.angle = tree.angle + Decimal(str(math.degrees(dtheta)))

        if (
            tree.center_x < CENTER_MIN
            or tree.center_x > CENTER_MAX
            or tree.center_y < CENTER_MIN
            or tree.center_y > CENTER_MAX
        ):
            restore_states(trees, states + [(idx, state)])
            return None

        tree.update_polygon()
        states.append((idx, state))

    return states
def simulated_annealing_for_n(
    n: int,
    settings: SASettings,
    initial_layout: list[ChristmasTree] | None = None,
):
    """
    Run SA for a specific puzzle size n using the provided settings.

    Returns:
        best_trees: list[ChristmasTree]
        best_cost: float
    """
    # Initial layout
    if initial_layout is not None:
        current = [deepcopy(t) for t in initial_layout]
    else:
        current = choose_initial_layout(n)
    compact_layout(current)

    current_cost = layout_cost(current)
    best = deepcopy(current)
    best_cost = current_cost

    max_iter = settings.iter_small if n < N_THRESHOLD_LARGE else settings.iter_large
    compact_every = max(25, max_iter // 20)
    temperature = T_START
    cooling_multiplier = (T_END / T_START) ** (1 / max(1, max_iter))
    recent_accepts = 0
    recent_attempts = 0
    since_improvement = 0
    move_types = list(MOVE_WEIGHTS.keys())
    move_weights = list(MOVE_WEIGHTS.values())

    for step in range(max_iter):
        scale = max(0.1, min(1.0, temperature / T_START))
        move_type = random.choices(move_types, weights=move_weights, k=1)[0]
        move_states = None

        if move_type == "single":
            dx = random_small_delta(STEP_XY * scale)
            dy = random_small_delta(STEP_XY * scale)
            dtheta = random_small_delta(STEP_ANGLE * scale)
            idx = random.randrange(len(current))

            if random.random() < GUIDED_MOVE_PROB:
                guided_vector = compute_guided_move(current, idx)
                if guided_vector is not None:
                    dx, dy = guided_vector
                    dtheta = 0.0

            move_states = apply_single_move(current, idx, dx, dy, dtheta)
        elif move_type == "swap":
            move_states = apply_swap_move(current)
        elif move_type == "cluster":
            move_states = apply_cluster_rotate_move(current)

        if move_states is None:
            continue

        # Occasionally re-anchor layout (cheap compaction)
        if step % compact_every == 0:
            compact_layout(current)

        new_cost = layout_cost(current)
        delta = new_cost - current_cost

        accept = False
        if delta <= 0:
            accept = True
        else:
            if temperature > 0:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    accept = True

        if accept:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best = deepcopy(current)
                since_improvement = 0
            else:
                since_improvement += 1
            recent_accepts += 1
        else:
            restore_states(current, move_states)
            since_improvement += 1

        recent_attempts += 1

        if (step + 1) % ACCEPT_WINDOW == 0:
            rate = recent_accepts / max(1, recent_attempts)
            temperature = adjust_temperature(temperature, rate)
            recent_accepts = 0
            recent_attempts = 0

        temperature = max(T_END, temperature * cooling_multiplier)

        if since_improvement >= KICK_THRESHOLD:
            kick_layout(current)
            current_cost = layout_cost(current)
            since_improvement = 0

    return best, best_cost


# ----------------------------------------------------------------------
# Multiprocessing worker & CSV writer
# ----------------------------------------------------------------------


def sa_worker(n: int, settings: SASettings):
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

    for restart in range(settings.max_restarts):
        seed = RNG_SEED + settings.seed_offset + n * 1000 + restart
        random.seed(seed)

        trees, cost = simulated_annealing_for_n(n, settings)

        if cost < best_cost:
            best_cost = cost
            best_trees = trees

        if restart == 0 and best_cost <= settings.good_threshold:
            break

    assert best_trees is not None
    layout = [(str(t.center_x), str(t.center_y), str(t.angle)) for t in best_trees]
    return n, layout, best_cost


def run_sa_batch(
    ns: list[int],
    settings: SASettings,
    results: dict[int, dict],
    predicate,
):
    pending = [n for n in ns if predicate(n, results)]
    if not pending:
        console.print(
            f"[yellow]No pending puzzles for {settings.label} SA pass.[/yellow]"
        )
        return

    console.print(
        f"[bold cyan]{settings.label.title()} SA pass: {len(pending)} puzzles[/bold cyan]"
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
            task = progress.add_task(f"{settings.label.title()} SA", total=len(pending))

            futures = {executor.submit(sa_worker, n, settings): n for n in pending}

            for fut in as_completed(futures):
                n = futures[fut]
                try:
                    n_ret, layout, best_cost = fut.result()
                except Exception as e:
                    console.print(f"[red]Worker for n={n} failed: {e}[/red]")
                    progress.update(task, advance=1)
                    continue

                record_result(n_ret, layout, best_cost, settings, results)
                progress.update(
                    task,
                    advance=1,
                    description=f"{settings.label.title()} SA (last n={n_ret})",
                )
                console.log(
                    f"{settings.label.title()} SA finished n={n_ret:3d}, best_cost={best_cost:.6f}"
                )

    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Ctrl+C detected. Cancelling workers for this pass...[/yellow]"
        )
        executor.shutdown(wait=False, cancel_futures=True)
        raise SystemExit(1)
    else:
        executor.shutdown(wait=True)


def write_submission(results: dict[int, dict], path: str = "submission_sa.csv"):
    """
    Write Kaggle-style submission CSV from compressed layouts.

    results must contain entries for every n in 1..MAX_N.
    """
    missing = [n for n in range(1, MAX_N + 1) if n not in results]
    if missing:
        console.print(
            f"[yellow]Skipping submission write; missing results for n={missing[:5]}...[/yellow]"
        )
        return

    rows = []
    for n in range(1, MAX_N + 1):
        layout = results[n]["layout"]
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


def summarize_results(results: dict[int, dict]) -> None:
    if not results:
        console.print("[red]No SA results available yet.[/red]")
        return

    completed = sorted(results.keys())
    costs = [results[n]["best_cost"] for n in completed]
    min_cost = min(costs)
    max_cost = max(costs)
    med_cost = median(costs)
    total_cost = sum(costs)

    console.print(
        f"\n[bold green]Collected SA results for {len(completed)}/{MAX_N} puzzles."
        f" Total cost over completed set: {total_cost:.6f}[/bold green]"
    )
    console.print(
        f"[cyan]Cost summary:[/cyan] min={min_cost:.6f}, median={med_cost:.6f}, max={max_cost:.6f}"
    )

    worst = sorted(results.items(), key=lambda kv: kv[1]["best_cost"], reverse=True)[
        :10
    ]
    console.print("[magenta]Worst n by best_cost:[/magenta]")
    for n, entry in worst:
        console.print(f"  n={n:3d}, best_cost={entry['best_cost']:.6f}")

    if len(completed) == MAX_N:
        write_submission(results, "submission_sa.csv")
    else:
        missing = [n for n in range(1, MAX_N + 1) if n not in results]
        console.print(
            f"[yellow]Submission not written; still missing n={missing[:10]} (total {len(missing)}).[/yellow]"
        )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    ns = list(range(1, MAX_N + 1))
    results = load_partial_results()

    if results:
        console.print(
            f"[bold cyan]Loaded {len(results)} existing SA results. Skipping completed n.[/bold cyan]"
        )
    else:
        console.print(
            "[bold cyan]No partial results found. Starting fresh run.[/bold cyan]"
        )

    run_sa_batch(ns, LIGHT_SETTINGS, results, needs_light_run)
    run_sa_batch(ns, HEAVY_SETTINGS, results, needs_heavy_run)

    summarize_results(results)


if __name__ == "__main__":
    main()
