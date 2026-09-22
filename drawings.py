"""Drawing extraction and clustering for PDF pages."""
# The module provides utilities for detecting, extracting, and saving drawing
# elements (e.g. figures, diagrams) from PDF pages. It clusters nearby
# drawing components, filters them based on size, margin, and overlap rules,
# and finally saves each cluster as an image file.


import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pymupdf

from layout import BoundingBox, passes_size_filter
from overlap import (
    calculate_bbox_distance,
    calculate_bbox_overlap,
    check_margin_violation,
    is_high_overlap_similar_size,
)
from post_process import count_non_whitespace_chars


@dataclass
class DrawingInfo:
    """Information about an extracted drawing cluster."""
    # Attributes store metadata needed for later saving and reporting.
    index: int                                 # Cluster index (0‑based)
    page_num: int                              # PDF page number (1‑based)
    saved: bool                                # Whether the drawing was saved
    filename: Optional[str]                    # Name of the saved file (if any)
    filepath: Optional[str]                    # Full path to the saved file (if any)
    resolution: List[int]                      # Width/height in pixels (rendered)
    skipped_reason: Optional[str]              # Why the drawing was skipped (if any)
    save_error: Optional[str]                  # Exception message if saving failed
    bbox: BoundingBox                          # Bounding box of the cluster
    original_clusters_merged: int = 1          # How many original clusters were merged
    includes_small_text: bool = False          # Whether the cluster contains small text
    total_items: int = 0                       # Total items (drawings + text) in cluster
    text_blocks_included: int = 0              # Number of text blocks included


class DrawingExtractor:
    """Extract and save drawing clusters from PDF pages."""

    def __init__(self, config, pdf_filename: str, drawing_dir: str):
        # Store configuration and file paths.
        self.config = config
        self.pdf_filename = pdf_filename
        self.drawing_dir = drawing_dir

    def extract_drawings(
        self,
        page: pymupdf.Page,
        page_num: int,
        reference_bboxes: Optional[Dict[str, List[Tuple[float, float, float, float]]]] = None,
        valid_drawing_count: int = 0,
        stats: Optional[Any] = None,
    ) -> Tuple[List[DrawingInfo], int, int]:
        """Extract drawing clusters from a page.
        Returns:
            A list of :class:`DrawingInfo` objects, the updated drawing count,
            and the number of small text blocks merged.
        """
        drawing_list: List[DrawingInfo] = []
        small_text_blocks_merged = 0

        # Retrieve already clustered drawing groups from the page.
        drawing_clusters = list(page.cluster_drawings()) if page.cluster_drawings() else []

        cluster_items: List[Dict[str, Any]] = [
            {
                "bbox": tuple(float(v) for v in bbox),
                "type": "drawing",
            }
            for bbox in drawing_clusters
        ]

        # Small text blocks are first-class merge items, just like drawing clusters.
        if self.config.include_small_text_blocks:
            blocks = page.get_text("blocks")
            small_blocks = self._get_small_text_blocks(blocks)
            cluster_items.extend(
                {
                    "bbox": tuple(float(v) for v in bbox),
                    "type": "small_text",
                }
                for bbox in small_blocks
            )

        # Early exit if no clusters remain after filtering.
        if not cluster_items:
            return drawing_list, valid_drawing_count, 0

        # Merge nearby drawing clusters and small text blocks together.
        merged_clusters = self._merge_nearby_clusters(cluster_items)

        # Cache expensive page calls once per page — reused across all clusters.
        page_drawings = page.get_drawings()

        # Pre-compute max allowed dimensions (80% of page width/height)
        max_width = page.rect.width * 0.8
        max_height = page.rect.height * 0.8

        # Process each merged cluster.
        for cluster_info in merged_clusters:
            cluster_bbox = cluster_info["bbox"]
            bbox_obj = BoundingBox.from_tuple(cluster_bbox)

            # Skip clusters that violate the configured margin.
            if check_margin_violation(bbox_obj, page.rect.width, page.rect.height, self.config):
                if stats is not None:
                    stats.drawings_skipped_margin += 1
                continue

            # NEW: Skip if drawing exceeds 80% of page width or height
            if bbox_obj.width > max_width or bbox_obj.height > max_height:
                if stats is not None:
                    stats.drawings_skipped_size += 1
                continue

            # Count drawing rectangles inside the final merged bbox.
            # Small text blocks are counted from the union-find membership
            drawing_items = self._count_drawing_items_cached(page_drawings, cluster_bbox)
            text_blocks = int(cluster_info["text_block_count"])

            # Enforce a minimum number of drawing items per cluster.
            if drawing_items < self.config.min_items_per_cluster:
                if stats is not None:
                    stats.drawings_skipped_min_items += 1
                continue

            # Apply size constraints in pixels (derived from DPI conversion).
            if self.config.min_size_px > 0:
                width_px = int(bbox_obj.width / 72 * 96)
                height_px = int(bbox_obj.height / 72 * 96)
                min_dimension = min(width_px, height_px)
                max_dimension = max(width_px, height_px)
                if max_dimension < self.config.min_size_px or min_dimension < self.config.min_size_px / 4:
                    if stats is not None:
                        stats.drawings_skipped_size += 1
                    continue

            # When reference objects (tables, images, etc.) are provided, reject clusters that overlap them.
            if reference_bboxes is not None:
                skipped_reasons = self._check_overlap(cluster_bbox, reference_bboxes)
                if skipped_reasons:
                    if stats is not None:
                        stats.drawings_skipped_overlap += 1
                    continue

            # Respect a global maximum number of drawings per page.
            if self.config.max_items_per_page > 0 and valid_drawing_count >= self.config.max_items_per_page:
                if stats is not None:
                    stats.drawings_skipped_limit += 1
                continue

            # If we reach this point the cluster is valid.
            valid_drawing_count += 1
            drawing_info = DrawingInfo(
                index=valid_drawing_count - 1,
                page_num=page_num,
                saved=True,
                filename=f"{self.pdf_filename}_page_{page_num:04d}_drawing_{valid_drawing_count:02d}.png",
                filepath=None,
                resolution=[int(bbox_obj.width / 72 * 96), int(bbox_obj.height / 72 * 96)],
                skipped_reason=None,
                save_error=None,
                bbox=bbox_obj,
                original_clusters_merged=cluster_info["cluster_count"],
                includes_small_text=text_blocks > 0,
                total_items=drawing_items + text_blocks,
                text_blocks_included=text_blocks,
            )
            drawing_list.append(drawing_info)

        # Count only small text blocks that were actually included in valid clusters.
        small_text_blocks_merged = sum(d.text_blocks_included for d in drawing_list)
        return drawing_list, valid_drawing_count, small_text_blocks_merged

    

    def save_all(self, drawings: List[DrawingInfo], page: pymupdf.Page):
        """Commit drawings to disk using the configured crop padding."""
        if any(drawing.saved for drawing in drawings):
            os.makedirs(self.drawing_dir, exist_ok=True)
        for drawing in drawings:
            if drawing.saved:
                # Resolve the full file path and actually write the image.
                drawing.filepath = os.path.join(self.drawing_dir, drawing.filename)
                self._save_drawing_to_disk(drawing, page)

    def _save_drawing_to_disk(self, drawing: DrawingInfo, page: pymupdf.Page):
        """Perform actual I/O, applying padding only at crop time."""
        try:
            # Compute crop rectangle with optional padding, staying inside page bounds.
            clip_x0 = max(0, drawing.bbox.x0 - self.config.bbox_padding)
            clip_y0 = max(0, drawing.bbox.y0 - self.config.bbox_padding)
            clip_x1 = min(page.rect.width, drawing.bbox.x1 + self.config.bbox_padding)
            clip_y1 = min(page.rect.height, drawing.bbox.y1 + self.config.bbox_padding)

            # Render the page region to a pixmap at 150 DPI.
            pix = page.get_pixmap(clip=(clip_x0, clip_y0, clip_x1, clip_y1), dpi=150)
            pix.save(drawing.filepath)
        except Exception as exc:
            # Mark as unsaved and store the error for later diagnostics.
            drawing.saved = False
            drawing.save_error = str(exc)

    def _get_small_text_blocks(self, blocks: List[Tuple]) -> List[Tuple[float, float, float, float]]:
        """Extract small text blocks that contain European/ASCII content.
        Only blocks whose non‑whitespace character count does not exceed the
        configured ``small_text_max_chars`` and whose area is at least 10 000 px²
        are kept.
        """
        small_blocks = []
        for block in blocks:
            if not block or len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            bbox_tuple = (x0, y0, x1, y1)

            if text and count_non_whitespace_chars(text) <= self.config.small_text_max_chars:
                area = (x1 - x0) * (y1 - y0)
                if area <= 500 and self._is_european_text(text): # min area used to avoiding large textblock becomming a drawing
                    small_blocks.append(bbox_tuple)
        return small_blocks


    def _is_european_text(self, text: str) -> bool:
        """Return ``True`` if *text* consists mainly of European characters.
        This includes Latin, Greek, Cyrillic, and ASCII digits.
        Text is considered "mainly" European if over 50% of non-whitespace
        characters fall within the defined European ranges.
        """
        if not text:
            return False

        european_ranges = [
            (0x0041, 0x007A),   # Basic Latin (A‑Z, a‑z)
            (0x00C0, 0x00FF),   # Latin‑1 Supplement
            (0x0100, 0x024F),   # Latin Extended‑A/B
            (0x1E00, 0x1EFF),   # Latin Extended Additional
            (0x0370, 0x03FF),   # Greek
            (0x0400, 0x04FF),   # Cyrillic
            (0x0500, 0x052F),   # Cyrillic Supplement
            (0x2C60, 0x2C7F),   # Cyrillic Extended
            (0xA640, 0xA69F),   # Cyrillic Extended‑A
        ]

        european_count = 0
        for char in text:
            code = ord(char)
            for start, end in european_ranges:
                if start <= code <= end:
                    european_count += 1
                    break
            else:
                # Also accept ASCII digits.
                if 0x0030 <= code <= 0x0039:
                    european_count += 1

        # Consider text "mainly" European if over 50% of characters match
        return european_count / len(text) > 0.5


    def _merge_nearby_clusters(self, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge spatially close drawing clusters and small text blocks.

        Each input cluster is expected to have the form:

            {
                "bbox": (x0, y0, x1, y1),
                "type": "drawing" | "small_text",
            }

        The method uses union-find so that nearby drawing clusters and small
        text blocks are merged into one final drawing area.
        """
        if not clusters:
            return []

        n = len(clusters)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path halving
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        bboxes: List[Tuple[float, float, float, float]] = [
            tuple(float(v) for v in cluster["bbox"]) for cluster in clusters
        ]
        bbox_objs: List[BoundingBox] = [BoundingBox.from_tuple(bbox) for bbox in bboxes]

        # Sort indices by x0 then y0 for spatial locality.
        sorted_indices = sorted(range(n), key=lambda i: (bboxes[i][0], bboxes[i][1]))

        tolerance = self.config.cluster_merge_tolerance

        # Linear scan over spatially nearby candidates.
        for i_idx in range(n):
            i_id = sorted_indices[i_idx]
            ix0, iy0, ix1, _ = bboxes[i_id]

            for j_idx in range(i_idx + 1, n):
                j_id = sorted_indices[j_idx]
                jx0, jy0, jx1, _ = bboxes[j_id]

                # Early break: if j's x0 exceeds i's x1 + tolerance, no further
                # clusters (sorted by x0) can be within distance.
                if jx0 > ix1 + tolerance:
                    break

                # Check for proximity OR containment
                should_merge = False

                # 1. Standard proximity check
                distance = calculate_bbox_distance(bbox_objs[i_id], bbox_objs[j_id])
                if distance <= tolerance:
                    should_merge = True

                # 2. Containment check: Is center of i inside j, or center of j inside i?
                if not should_merge:
                    # Center of i in j?
                    i_center_x = (bboxes[i_id][0] + bboxes[i_id][2]) / 2
                    i_center_y = (bboxes[i_id][1] + bboxes[i_id][3]) / 2
                    if (jx0 <= i_center_x <= jx1) and (jy0 <= i_center_y <= bboxes[j_id][3]):
                        should_merge = True

                    # Center of j in i?
                    if not should_merge:
                        j_center_x = (bboxes[j_id][0] + bboxes[j_id][2]) / 2
                        j_center_y = (bboxes[j_id][1] + bboxes[j_id][3]) / 2
                        if (ix0 <= j_center_x <= ix1) and (iy0 <= j_center_y <= bboxes[i_id][3]):
                            should_merge = True

                if should_merge:
                    union(i_id, j_id)

        # Group indices by their root parent.
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        result: List[Dict[str, Any]] = []

        for group_indices in groups.values():
            x0_min = min(bboxes[i][0] for i in group_indices)
            y0_min = min(bboxes[i][1] for i in group_indices)
            x1_max = max(bboxes[i][2] for i in group_indices)
            y1_max = max(bboxes[i][3] for i in group_indices)

            text_block_count = sum(
                1
                for i in group_indices
                if clusters[i].get("type") == "small_text"
            )
            drawing_cluster_count = len(group_indices) - text_block_count

            result.append(
                {
                    "bbox": (x0_min, y0_min, x1_max, y1_max),
                    "original_indices": group_indices,
                    "cluster_count": len(group_indices),
                    "drawing_cluster_count": drawing_cluster_count,
                    "text_block_count": text_block_count,
                }
            )

        return result




    def _count_drawing_items_cached(
        self,
        drawings: List[Any],
        cluster_bbox: Tuple[float, float, float, float],
    ) -> int:
        """Count drawing rectangles completely inside *cluster_bbox*."""
        x0, y0, x1, y1 = cluster_bbox

        rect_count = 0

        for drawing in drawings:
            if "rect" not in drawing:
                continue

            dr = drawing["rect"]

            if hasattr(dr, "x0"):
                dr = (float(dr.x0), float(dr.y0), float(dr.x1), float(dr.y1))
            elif isinstance(dr, (tuple, list)):
                dr = tuple(float(v) for v in dr)
            else:
                continue

            # Simple containment test.
            if dr[0] >= x0 and dr[1] >= y0 and dr[2] <= x1 and dr[3] <= y1:
                rect_count += 1

        return rect_count



    def _check_overlap(
        self,
        cluster_bbox: Tuple[float, float, float, float],
        reference_bboxes: Dict[str, List[Tuple[float, float, float, float]]],
    ) -> List[Dict[str, Any]]:
        """Determine whether *cluster_bbox* overlaps with tables, images, or text blocks.
        If an overlap is found, a reason is added to ``skipped_reasons`` and the
        cluster will be discarded.
        """
        bbox = BoundingBox.from_tuple(cluster_bbox)
        skipped_reasons: List[Dict[str, Any]] = []

        # Examine each category of reference objects.
        for key in ["tables", "images", "text_blocks"]:
            if key not in reference_bboxes:
                continue

            for ref_bbox in reference_bboxes[key]:
                ref_box = BoundingBox.from_tuple(ref_bbox)
                overlap_pct, _, has_overlap = calculate_bbox_overlap(bbox, ref_box)
                if not has_overlap:
                    continue

                # Table overlap takes precedence if the drawing's centre lies inside it.
                if key == "tables":
                    if (
                        ref_box.x0 <= bbox.center_x <= ref_box.x1
                        and ref_box.y0 <= bbox.center_y <= ref_box.y1
                    ):
                        skipped_reasons.append(
                            {
                                "type": key,
                                "overlap_percentage": overlap_pct,
                                "reason": "Table takes precedence (drawing center inside table)",
                            }
                        )
                        break
                    # Or if the overlap exceeds a generous threshold.
                    elif overlap_pct >= 0.8:
                        skipped_reasons.append(
                            {
                                "type": key,
                                "overlap_percentage": overlap_pct,
                                "reason": "Table takes precedence (>50% overlap)",
                            }
                        )
                        break
                # For images and text blocks, use a stricter high‑overlap check.
                elif key in ("images", "text_blocks"):
                    is_overlapping, _ = is_high_overlap_similar_size(
                        bbox,
                        ref_box,
                        overlap_threshold=0.8,
                        size_tolerance=0.2,
                    )
                    if is_overlapping:
                        skipped_reasons.append(
                            {
                                "type": key,
                                "overlap_percentage": overlap_pct,
                                "reason": f"{key} overlap detected",
                            }
                        )
                        break

            # If a table caused a skip, no need to evaluate remaining categories.
            if skipped_reasons and skipped_reasons[0]["type"] == "tables":
                break

        return skipped_reasons

