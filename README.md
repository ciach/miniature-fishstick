# Christmas Tree Packing – Problem Description & Solution Options

## 1. Problem Description

We consider a 2D irregular packing problem derived from the **Kaggle Santa 2025 – Christmas Tree Packing** setting.

- We have a single fixed polygonal shape (a “Christmas tree”).
- For each puzzle size `n` (from 1 to `MAX_N`, typically up to 200), we must place `n` copies of this shape:
  - Each copy can be translated in the plane.
  - Each copy can be rotated by an arbitrary angle.
- The layout must be **collision-free** (no overlapping interiors of the trees).
- The trees must stay within a bounded region (in this codebase, their *centers* are constrained to `[-100, 100] × [-100, 100]`).

The goal is to produce, for each `n`, a layout (set of `(x, y, angle)` triples) that gives a **low layout cost**, which is the proxy objective used in the geometry module (`layout_cost`).

Smaller cost is better and corresponds to “more compact, better packed” arrangements.

---

## 2. Formal Problem Statement

For a given integer `n ≥ 1`, let:

- `P` be the fixed polygon representing a single tree in some local coordinate system.
- A placement is defined by `(x_i, y_i, θ_i)` for tree `i`, representing:
  - Translation by `(x_i, y_i)`
  - Rotation by angle `θ_i` (in degrees) about the tree’s reference point (e.g. its center).

We seek to choose placements `{(x_i, y_i, θ_i)}_{i=1..n}` such that:

1. **Non-overlap constraint**

   Let `T_i` be the polygon `P` transformed by `(x_i, y_i, θ_i)`.

   For all `i ≠ j`:
   - Interiors do not overlap:
     \[
     \operatorname{interior}(T_i) \cap \operatorname{interior}(T_j) = \varnothing
     \]
   - Touching at edges or vertices may be allowed, depending on how intersection vs touch is treated.

2. **Bounding region constraint**

   Each tree center must stay within a fixed box:
   \[
   x_i, y_i \in [CENTER\_MIN, CENTER\_MAX]
   \]
   where in this implementation:
   - `CENTER_MIN = -100`
   - `CENTER_MAX = 100`

3. **Objective**

   Minimize a scalar objective:
   \[
   \text{cost} = \text{layout\_cost}(\{T_i\})
   \]
   where `layout_cost` is a custom function that typically reflects:
   - Size of the enclosing square / bounding box.
   - Possibly penalties or scaling related to Kaggle’s scoring.

---

## 3. Constraints & Practical Considerations

- **Geometry & precision**
  - Polygons are handled with Shapely.
  - Coordinates are stored as `Decimal` in the `ChristmasTree` class for better numeric stability.
  - Polygons must be updated whenever a tree’s center or angle changes.

- **Search space**
  - The search space is continuous in `x`, `y`, `θ` and very high-dimensional (`3n` parameters).
  - Exact optimal packing is intractable; we rely on heuristics and metaheuristics.

- **Feasibility**
  - Overlaps are disallowed; any move that pushes a tree outside the allowed center range is immediately rejected.

---

## 4. Objective Function

The exact implementation of `layout_cost` lives in the `geometry` module. Conceptually:

- It aggregates geometric information about the current layout into a single numeric score.
- Typical components for such a function include:
  - Width/height of the axis-aligned bounding box of all trees.
  - Possibly the side length of the minimal square enclosing all trees.
- The Kaggle score is monotonic in this cost: **lower `layout_cost` means a better solution**.

This script treats `layout_cost` as a black-box function:
- Any candidate layout → single float cost.
- Used inside greedy initialization, local modifications, and simulated annealing acceptance decisions.

---

## 5. Solution Approaches (High-Level)

Packing irregular shapes is hard, so we rely on heuristics. Below are common strategies, including what is used or planned in this project.

### 5.1 Greedy Constructive Methods

**Idea:**
Build a layout by placing trees one by one in a “reasonable” position.

Examples:
- Place the first tree at the origin.
- For each next tree:
  - Choose a direction (random angle or structured pattern).
  - Start far away, move inward until collision, then back off slightly.
  - Optionally compact or recenter the whole layout after a batch of placements.

Pros:
- Very fast.
- Always produces a feasible starting solution.

Cons:
- Usually far from optimal.
- Easily gets stuck in “locally crowded” layouts.

In `optimizer_sa_parallel_imp.py`, `greedy_initialize_single_n` implements this type of initializer.

---

### 5.2 Local Search / Hill Climbing

**Idea:**
Start from an initial layout and iteratively try small random changes:

- Randomly pick a tree.
- Apply a small perturbation in position and/or angle.
- If cost improves and constraints are satisfied, accept the move.
- Otherwise revert.

Pros:
- Easy to implement.
- Improves over naive greedy quickly.

Cons:
- Gets stuck in local minima.
- No explicit mechanism to escape poor basins.

This logic is effectively embedded inside simulated annealing when temperature is very low.

---

### 5.3 Simulated Annealing (Current Script)

**Idea:**
Like hill climbing, but allow *worse* moves with a probability that decreases over time (“temperature cooling”).

Algorithm sketch for given `n`:

1. **Initialization**
   - Build an initial layout via `greedy_initialize_single_n`.
   - Compact the layout (`compact_layout`) so it sits near the origin.
   - Compute `current_cost` and set `best = current`.

2. **SA loop**
   - For each iteration `step`:
     - Compute temperature `T` via a cooling schedule between `T_START` and `T_END`.
     - Propose a random move for a randomly chosen tree:
       - Small `dx`, `dy`, `dθ` scaled by the current progress in the schedule.
       - Reject immediately if the moved center leaves the `[CENTER_MIN, CENTER_MAX]` box.
     - Occasionally re-compact the layout (cheap centering).
     - Evaluate new cost:
       - If `delta = new_cost - current_cost <= 0`, accept (improvement).
       - Else accept with probability `exp(-delta / T)`.
     - Track the best cost and layout seen.

3. **Multi-start**
   - Run SA multiple times with different seeds.
   - Early-stop if the first run achieves a “good enough” threshold.

The script implements two configurations:

- **Light settings**
  - Fewer iterations / restarts.
  - Used as a first pass for all `n`.
- **Heavy settings**
  - More iterations / restarts.
  - Only used if the light pass result is still above a threshold (`HEAVY_RERUN_THRESHOLD`).

Pros:
- Can escape local minima.
- Reasonably generic and robust.

Cons:
- Computationally expensive (requires many cost evaluations).
- Sensitive to cooling schedule and step sizes.

---

### 5.4 Genetic Algorithms / Evolutionary Search

**Idea:**
Treat each layout as an individual in a population:

- A “chromosome” encodes all `(x, y, θ)` for all trees.
- Evaluate fitness using `layout_cost`.
- Apply selection, crossover, and mutation to produce new generations.
- Optionally use local search (e.g. a few SA steps) as a “mutation” or local improvement operator.

Pros:
- Explores many regions of the search space in parallel.
- Crossovers can combine good partial patterns.

Cons:
- Representation and crossover need to be carefully designed.
- Expensive if each evaluation is heavy (Shapely + many collisions).

A GA is already used in other scripts in this project (e.g. `ga_constructive_single.py`), and can be combined with SA as a hybrid metaheuristic.

---

### 5.5 Pattern-Based / Analytic Layouts

**Idea:**
Instead of arbitrary placements, search over structured patterns:

- Regular grids (axis-aligned).
- Staggered / hex-like patterns.
- Repeating “tiles” of few trees that interlock well.
- Parameterized families:
  - Shift between rows.
  - Rotation of alternating rows.
  - Distances `dx`, `dy`, row offsets, global angle, etc.

Workflow:

1. Define a parameterized pattern for tree positions.
2. Optimize over pattern parameters (grid search, random search, or small local search).
3. Optionally refine the best patterns with SA or GA.

Pros:
- Very fast to search.
- Often finds surprisingly good packings for medium `n`.

Cons:
- Limited by the expressiveness of the pattern family.
- May struggle for awkward `n` where structured patterns are suboptimal.

---

### 5.6 Advanced Irregular Packing Methods

For more aggressive optimization, one can use more geometric knowledge:

- **Penetration depth / separation-based search**
  - Given overlapping polygons, compute minimal translations or rotations to separate them.
  - Use these “collision resolution” vectors as guided moves instead of random perturbations.
- **Guided local search**
  - Penalize frequently colliding pairs or regions.
  - Encourage exploration of less congested configurations.

These methods can be combined with SA or GA:
- Use geometric information to propose smarter moves rather than purely random deltas.

---

## 6. Current Script (`optimizer_sa_parallel_imp.py`) – Summary

The script provides:

1. **Partial results management**
   - Loads/saves `sa_partial_results.json`.
   - Supports resuming runs, skipping already solved `n`.
   - Tracks whether light/heavy SA was applied.

2. **Simulated annealing engine**
   - `greedy_initialize_single_n(n)` builds initial placements.
   - `compact_layout(trees)` recenters configurations.
   - `simulated_annealing_for_n(n, settings)` runs the SA loop for a single `n`.

3. **Parallel execution**
   - `sa_worker(n, settings)` runs multi-start SA for one `n`.
   - `run_sa_batch(ns, settings, results, predicate)` uses a `ProcessPoolExecutor` to solve many `n` in parallel, with progress reporting via `rich`.

4. **Submission generation**
   - `write_submission(results, path)` converts the best layouts into the Kaggle submission CSV format.
   - `summarize_results(results)` prints stats and writes the submission when all `n` are complete.

---

## 7. Possible Extensions / Improvements

Potential future improvements to this approach include:

- **Better initial layouts**
  - Use pattern-based placements or outputs from other heuristics as starting solutions for SA.
  - Use simple pairwise “docking” to align trees more tightly from the start.

- **Richer move set**
  - Separate translation vs rotation moves and tune their frequencies.
  - Introduce larger “jump” moves occasionally to escape deep local minima.

- **Adaptive schedules**
  - Tune `T_START`, `T_END`, and cooling schedule based on observed acceptance rate.
  - Increase iterations for “hard” `n` where progress is still happening near the end.

- **Hybrid algorithms**
  - GA to explore high-level patterns, with SA or local search as local improvement.
  - Large neighborhood search (LNS): periodically remove and re-insert a subset of trees with a constructive heuristic.

- **Incremental geometry / cost**
  - Cache geometric data and update only affected parts when moving a single tree.
  - Reduce the cost of each move evaluation.

These directions aim to improve either **solution quality**, **runtime efficiency**, or both, while staying compatible with the current geometry and result-persistence structure.
