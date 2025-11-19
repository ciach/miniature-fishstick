import math

from geometry import ChristmasTree, has_overlap
from optimizer_sa_parallel_imp import (
    ACCEPT_RATE_HIGH,
    ACCEPT_RATE_LOW,
    T_START,
    adjust_temperature,
    compute_guided_move,
    hex_grid_initialize_single_n,
    kick_layout,
    pattern_initialize_single_n,
    repulsion_initialize_single_n,
)


def test_guided_move_points_away_from_overlap():
    t1 = ChristmasTree(center_x="0", center_y="0", angle="0")
    t2 = ChristmasTree(center_x="0.3", center_y="0", angle="0")
    trees = [t1, t2]

    move = compute_guided_move(trees, 0, step=0.4)
    assert move is not None
    dx, dy = move

    direction = float(t1.center_x) - float(t2.center_x)
    assert dx * direction > 0
    assert abs(dy) < 1e-6 or abs(dy) < abs(dx)


def test_pattern_initialization_respects_bounds():
    trees = pattern_initialize_single_n(20)
    assert len(trees) == 20
    assert not has_overlap(trees)
    for t in trees:
        assert -100.0 <= float(t.center_x) <= 100.0
        assert -100.0 <= float(t.center_y) <= 100.0


def test_hex_grid_initializer_valid():
    trees = hex_grid_initialize_single_n(25)
    assert len(trees) == 25
    assert not has_overlap(trees)


def test_repulsion_initializer_breaks_overlaps():
    trees = repulsion_initialize_single_n(15)
    if trees:
        assert len(trees) == 15
        assert not has_overlap(trees)


def test_kick_layout_moves_subset():
    trees = pattern_initialize_single_n(5)
    before = [(float(t.center_x), float(t.center_y)) for t in trees]
    kick_layout(trees)
    after = [(float(t.center_x), float(t.center_y)) for t in trees]
    assert before != after


def test_adjust_temperature_cools_and_reheats():
    cooled = adjust_temperature(0.8, ACCEPT_RATE_HIGH + 0.1)
    reheated = adjust_temperature(0.2, ACCEPT_RATE_LOW - 0.1)
    stable = adjust_temperature(0.7, (ACCEPT_RATE_HIGH + ACCEPT_RATE_LOW) / 2)

    assert cooled < 0.8
    assert reheated > 0.2
    assert math.isclose(stable, 0.7, rel_tol=1e-6)
    assert cooled >= 1e-3
    assert reheated <= T_START
