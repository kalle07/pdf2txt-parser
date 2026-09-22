"""Overlap, distance, and margin checks for parsed PDF regions."""

import math
from typing import Any, List, Optional, Tuple

from layout import BoundingBox


def calculate_bbox_overlap(
    bbox1: BoundingBox,
    bbox2: BoundingBox,
) -> Tuple[float, float, bool]:
    """Calculate Intersection‑over‑Union (IoU) and basic overlap info.

    Returns a tuple ``(iou, intersection_area, has_overlap)`` where:
        - ``iou`` is the IoU ratio (0‑1) of the two boxes,
        - ``intersection_area`` is the raw area of overlap,
        - ``has_overlap`` is ``True`` when the boxes actually intersect.
    """
    # Compute coordinates of the intersection rectangle
    inter_x0 = max(bbox1.x0, bbox2.x0)
    inter_y0 = max(bbox1.y0, bbox2.y0)
    inter_x1 = min(bbox1.x1, bbox2.x1)
    inter_y1 = min(bbox1.y1, bbox2.y1)

    # No overlap if the intersection is degenerate
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0, 0.0, False

    # Area of the overlapping region
    intersection_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)

    # Union area = area1 + area2 - intersection
    union_area = bbox1.area + bbox2.area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou, intersection_area, True


def calculate_bbox_distance(
    bbox1: BoundingBox,
    bbox2: BoundingBox,
) -> float:
    """Return the minimum Euclidean distance between two bounding boxes.

    The distance is measured from the nearest edges; if the boxes overlap
    the result is negative (but the function still returns a numeric value).
    """
    # Horizontal gap: positive when boxes are separated, negative when they overlap
    dx = max(bbox2.x0 - bbox1.x1, bbox1.x0 - bbox2.x1)
    # Vertical gap
    dy = max(bbox2.y0 - bbox1.y1, bbox1.y0 - bbox2.y1)

    if dx < 0 and dy < 0:
        # Boxes overlap in both dimensions → distance is the smaller (more negative) gap
        return min(dx, dy)
    if dx < 0:
        # Overlap only on X‑axis; distance is purely vertical
        return max(0, math.sqrt(dy ** 2))
    if dy < 0:
        # Overlap only on Y‑axis; distance is purely horizontal
        return max(0, math.sqrt(dx ** 2))
    # No overlap in either axis → Euclidean distance
    return max(0, math.sqrt(dx ** 2 + dy ** 2))


def check_margin_violation(
    bbox: BoundingBox,
    page_width: float,
    page_height: float,
    config: Any,
) -> bool:
    """Determine whether a box's centre lies outside the safe margin area.

    The margin configuration is controlled by ``config``; if
    ``config.enable_margin_check`` is ``False`` the function always returns ``False``.
    Otherwise the safe region is defined by the four margin percentages
    ``margin_left``, ``margin_right``, ``margin_top`` and ``margin_bottom``.
    """
    # Skip margin check if disabled in the configuration
    # Use getattr for defensive access since config type is Any
    enable_margin = getattr(config, "enable_margin_check", True)
    if not enable_margin:
        return False

    # Compute the safe rectangle limits using page dimensions and margin percentages
    safe_x0 = config.margin_left * page_width          # left edge
    safe_x1 = config.margin_right * page_width         # right edge
    safe_y0 = config.margin_top * page_height          # top edge
    safe_y1 = config.margin_bottom * page_height       # bottom edge

    # ``bbox.center_x`` / ``bbox.center_y`` are assumed to be pre‑computed centre coordinates
    is_safe = (
        bbox.center_x >= safe_x0
        and bbox.center_x <= safe_x1
        and bbox.center_y >= safe_y0
        and bbox.center_y <= safe_y1
    )
    # Return True when the centre is *outside* the safe area (i.e. a violation)
    return not is_safe


def is_high_overlap_similar_size(
    bbox1: BoundingBox,
    bbox2: BoundingBox,
    overlap_threshold: float = 0.8,
    size_tolerance: float = 0.2,
) -> Tuple[bool, float]:
    """Check whether two boxes have both high overlap and similar area.

    Returns a tuple ``(is_similar, overlap_pct)`` where:
        - ``is_similar`` is ``True`` when the overlap percentage meets or exceeds
          ``overlap_threshold`` *and* the relative size difference is within
          ``size_tolerance``.
        - ``overlap_pct`` is the raw IoU percentage (0‑100) of the overlap.
    """
    # Get overlap metrics; ``has_overlap`` tells us if they intersect at all
    overlap_pct, _, has_overlap = calculate_bbox_overlap(bbox1, bbox2)
    if not has_overlap or overlap_pct < overlap_threshold:
        return False, overlap_pct

    # Compare absolute areas; guard against division by zero
    area1 = bbox1.area
    area2 = bbox2.area
    if area2 == 0:
        return False, overlap_pct

    # Relative size difference (0‑1 range); smaller value means more similar
    size_diff_ratio = abs(area1 - area2) / max(area1, area2)
    if size_diff_ratio <= size_tolerance:
        return True, overlap_pct
    return False, overlap_pct


# PDF points to CSS/screen pixels:
# 72 points per PDF inch, 96 pixels per screen inch.
PX_TO_POINTS = 72.0 / 96.0


def _count_nonempty_lines(text: str) -> int:
    """Return the number of non-empty lines in a text block."""
    return len([line for line in text.splitlines() if line.strip()])


def is_small_text_block(
    text: str,
    max_lines: int = 2,
    max_chars: int = 250,
) -> bool:
    """
    Return True when a text block is small enough to be a table caption.

    A block is considered small when:
    - it has at most ``max_lines`` non-empty lines, and
    - its stripped text length is at most ``max_chars`` characters.
    """
    if text is None:
        return False

    stripped = text.strip()
    if not stripped:
        return False

    if len(stripped) > max_chars:
        return False

    if _count_nonempty_lines(stripped) > max_lines:
        return False

    return True


def horizontal_overlap_width(
    bbox1: BoundingBox,
    bbox2: BoundingBox,
) -> float:
    """Return the width of the horizontal intersection of two boxes."""
    inter_x0 = max(bbox1.x0, bbox2.x0)
    inter_x1 = min(bbox1.x1, bbox2.x1)
    return max(0.0, inter_x1 - inter_x0)


def vertical_gap_above_table(
    table_bbox: BoundingBox,
    text_bbox: BoundingBox,
) -> float:
    """
    Return the vertical gap between a text block and the top of a table.

    Positive value: text is above the table.
    Negative value: text overlaps below the table top.
    """
    return table_bbox.y0 - text_bbox.y1


def vertical_gap_below_table(
    table_bbox: BoundingBox,
    text_bbox: BoundingBox,
) -> float:
    """
    Return the vertical gap between the bottom of a table and a text block.

    Positive value: text is below the table.
    Negative value: text overlaps above the table bottom.
    """
    return text_bbox.y0 - table_bbox.y1


def is_close_above_table(
    table_bbox: BoundingBox,
    text_bbox: BoundingBox,
    max_distance_px: float = 10.0,
    max_overlap_px: float = 5.0,
    require_horizontal_overlap: bool = True,
) -> bool:
    """
    Return True when a text block is close above a table.

    The text block must:
    - be vertically close to the table top,
    - not overlap the table top by more than ``max_overlap_px``,
    - and, by default, horizontally overlap the table.
    """
    gap = vertical_gap_above_table(table_bbox, text_bbox)

    if gap > max_distance_px:
        return False

    if gap < -max_overlap_px:
        return False

    if require_horizontal_overlap:
        if horizontal_overlap_width(table_bbox, text_bbox) <= 0.0:
            return False

    return True


def is_close_below_table(
    table_bbox: BoundingBox,
    text_bbox: BoundingBox,
    max_distance_px: float = 10.0,
    max_overlap_px: float = 2.0,
    require_horizontal_overlap: bool = True,
) -> bool:
    """
    Return True when a text block is close below a table.

    The text block must:
    - be vertically close to the table bottom,
    - not overlap the table bottom by more than ``max_overlap_px``,
    - and, by default, horizontally overlap the table.
    """
    max_distance_pt = max_distance_px * PX_TO_POINTS
    max_overlap_pt = max_overlap_px * PX_TO_POINTS

    gap = vertical_gap_below_table(table_bbox, text_bbox)

    if gap > max_distance_pt:
        return False

    if gap < -max_overlap_pt:
        return False

    if require_horizontal_overlap:
        if horizontal_overlap_width(table_bbox, text_bbox) <= 0.0:
            return False

    return True


def combine_nonempty_texts(*parts: Optional[str]) -> Optional[str]:
    """
    Join non-empty text parts with newlines.

    Returns ``None`` when no part contains meaningful text.
    """
    cleaned = [part.strip() for part in parts if part and part.strip()]
    if not cleaned:
        return None
    return "\n".join(cleaned)

