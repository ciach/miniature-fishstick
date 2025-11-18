from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import List, Tuple

from shapely import affinity, touches
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Match Kaggle's precision setup
getcontext().prec = 25
scale_factor = Decimal("1e15")

# Bounds for center_x / center_y (Kaggle constraints)
CENTER_BOUNDS = (-100.0, 100.0)


@dataclass
class ChristmasTree:
    """
    Single Christmas tree with Kaggle's exact geometry.

    center_x, center_y, angle are in *unscaled* units:
      - x, y in [-100, 100]
      - angle in degrees

    Internally, the polygon is stored in scaled coordinates (× 1e15),
    just like the Kaggle starter, to avoid precision issues.
    """

    center_x: Decimal = Decimal("0")
    center_y: Decimal = Decimal("0")
    angle: Decimal = Decimal("0")
    polygon: Polygon | None = None

    def __post_init__(self) -> None:
        # Normalize inputs (support str / float / Decimal)
        self.center_x = Decimal(str(self.center_x))
        self.center_y = Decimal(str(self.center_y))
        self.angle = Decimal(str(self.angle))
        self.update_polygon()

    @staticmethod
    def _base_polygon_scaled() -> Polygon:
        """
        Build the base tree polygon at the origin, in scaled coordinates.

        This is copied from Kaggle's "Getting Started" code and is the
        source of truth for the tree shape.
        """
        trunk_w = Decimal("0.15")
        trunk_h = Decimal("0.2")
        base_w = Decimal("0.7")
        mid_w = Decimal("0.4")
        top_w = Decimal("0.25")
        tip_y = Decimal("0.8")
        tier_1_y = Decimal("0.5")
        tier_2_y = Decimal("0.25")
        base_y = Decimal("0.0")
        trunk_bottom_y = -trunk_h

        pts = [
            # Start at Tip
            (Decimal("0.0") * scale_factor, tip_y * scale_factor),
            # Right side - Top Tier
            (top_w / Decimal("2") * scale_factor, tier_1_y * scale_factor),
            (top_w / Decimal("4") * scale_factor, tier_1_y * scale_factor),
            # Right side - Middle Tier
            (mid_w / Decimal("2") * scale_factor, tier_2_y * scale_factor),
            (mid_w / Decimal("4") * scale_factor, tier_2_y * scale_factor),
            # Right side - Bottom Tier
            (base_w / Decimal("2") * scale_factor, base_y * scale_factor),
            # Right Trunk
            (trunk_w / Decimal("2") * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal("2") * scale_factor, trunk_bottom_y * scale_factor),
            # Left Trunk
            (-(trunk_w / Decimal("2")) * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal("2")) * scale_factor, base_y * scale_factor),
            # Left side - Bottom Tier
            (-(base_w / Decimal("2")) * scale_factor, base_y * scale_factor),
            # Left side - Middle Tier
            (-(mid_w / Decimal("4")) * scale_factor, tier_2_y * scale_factor),
            (-(mid_w / Decimal("2")) * scale_factor, tier_2_y * scale_factor),
            # Left side - Top Tier
            (-(top_w / Decimal("4")) * scale_factor, tier_1_y * scale_factor),
            (-(top_w / Decimal("2")) * scale_factor, tier_1_y * scale_factor),
        ]

        # Shapely wants floats
        pts_f = [(float(x), float(y)) for x, y in pts]
        return Polygon(pts_f)

    def update_polygon(self) -> None:
        """
        Rebuild self.polygon from current center_x, center_y, angle.

        Uses the same rotate + translate logic as the Kaggle starter.
        """
        base_poly = self._base_polygon_scaled()
        rotated = affinity.rotate(base_poly, float(self.angle), origin=(0.0, 0.0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor),
        )


# ---------- Helpers ----------


def placed_polygons(trees: List[ChristmasTree]) -> List[Polygon]:
    """Return a list of Shapely polygons for the current tree placements."""
    return [t.polygon for t in trees]


# ---------- Overlap test (Kaggle-style) ----------


def has_overlap(trees: List[ChristmasTree]) -> bool:
    """
    Return True if any pair of trees has a *real* overlap.

    Follows Kaggle logic:
      - collision if polygons intersect AND do NOT just touch.
    """
    polys = placed_polygons(trees)
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            p_i = polys[i]
            p_j = polys[j]
            if p_i.intersects(p_j) and not touches(p_i, p_j):
                return True
    return False


# ---------- Bounding square ----------


def bounding_square_side(
    trees: List[ChristmasTree],
) -> Tuple[Decimal, Tuple[Decimal, Decimal, Decimal, Decimal]]:
    """
    Compute the side length of the minimal axis-aligned square (in *unscaled* units)
    that contains all tree polygons.

    Returns:
        side_length (Decimal),
        (minx, miny, maxx, maxy) in unscaled coordinates.
    """
    if not trees:
        return Decimal("0"), (Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"))

    polys = placed_polygons(trees)
    bounds = unary_union(polys).bounds  # (minx, miny, maxx, maxy) in scaled floats

    minx = Decimal(str(bounds[0])) / scale_factor
    miny = Decimal(str(bounds[1])) / scale_factor
    maxx = Decimal(str(bounds[2])) / scale_factor
    maxy = Decimal(str(bounds[3])) / scale_factor

    width = maxx - minx
    height = maxy - miny
    side_length = max(width, height)

    return side_length, (minx, miny, maxx, maxy)


# ---------- Cost function ----------


def layout_cost(
    trees: List[ChristmasTree],
    overlap_penalty: Decimal | float = Decimal("1e6"),
    center_bounds: Tuple[float, float] = CENTER_BOUNDS,
) -> float:
    """
    Kaggle-style objective with penalties:

        base_cost = s^2 / n
        + overlap_penalty * total_overlap_area
        + overlap_penalty * (# of centers out of bounds)

    where:
      - s is the bounding square side (unscaled units)
      - n is number of trees

    The overlap area is computed in unscaled units^2 as well.
    """
    n = len(trees)
    if n == 0:
        return float("inf")

    overlap_penalty = Decimal(str(overlap_penalty))

    # Base cost from bounding square
    side, _ = bounding_square_side(trees)
    base_cost = (side * side) / Decimal(n)

    polys = placed_polygons(trees)
    total_overlap_area = Decimal("0")

    # Overlap penalty: sum of intersection areas (unscaled)
    for i in range(n):
        for j in range(i + 1, n):
            p_i = polys[i]
            p_j = polys[j]

            if p_i.intersects(p_j) and not touches(p_i, p_j):
                inter = p_i.intersection(p_j)
                # inter.area is in scaled^2; convert back to real units^2
                raw_area = Decimal(str(inter.area))
                real_area = raw_area / (scale_factor * scale_factor)
                total_overlap_area += real_area

    cost = base_cost + overlap_penalty * total_overlap_area

    # Center bounds penalty (keep centers in [-100, 100])
    lo, hi = center_bounds
    lo_dec = Decimal(str(lo))
    hi_dec = Decimal(str(hi))

    out_of_bounds = 0
    for t in trees:
        if (
            t.center_x < lo_dec
            or t.center_x > hi_dec
            or t.center_y < lo_dec
            or t.center_y > hi_dec
        ):
            out_of_bounds += 1

    if out_of_bounds > 0:
        cost += overlap_penalty * Decimal(out_of_bounds)

    return float(cost)


# ---------- Tiny sanity check ----------

if __name__ == "__main__":
    # Two trees: one at origin, one to the right
    t1 = ChristmasTree(center_x="0", center_y="0", angle="0")
    t2 = ChristmasTree(center_x="1.0", center_y="0", angle="30")

    trees = [t1, t2]

    print("Overlap:", has_overlap(trees))
    side, bbox = bounding_square_side(trees)
    print("Square side:", side)
    print("BBox:", bbox)
    print("Cost:", layout_cost(trees))
