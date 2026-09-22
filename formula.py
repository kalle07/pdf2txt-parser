"""Cambria Math formula-region extraction and rendering."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pymupdf

logger = logging.getLogger(__name__)

pymupdf.TOOLS.set_small_glyph_heights(True)

@dataclass
class FormulaInfo:
    """Metadata for one detected formula text block or cluster."""

    formula_num: int
    page_num: int
    block_indices: List[int]  # Original block indices merged into this cluster
    bbox: Tuple[float, float, float, float]
    filename: str
    saved: bool = False
    skipped_reason: Optional[str] = None
    char_count: int = 0
    cambria_math_char_count: int = 0
    cambria_math_ratio: float = 0.0
    save_error: Optional[str] = None


class FormulaExtractor:
    """Detect text blocks dominated by the Cambria Math font and render them."""

    def __init__(
        self,
        config: Any,
        pdf_filename: str,
        media_dir: str,
    ) -> None:
        self.config = config
        self.pdf_filename = pdf_filename
        self.media_dir = Path(media_dir)

    def extract_formulas(
        self,
        page: pymupdf.Page,
        page_num: int,
        exclude_bboxes: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    ) -> List[FormulaInfo]:
        """Detect and save formula blocks in document order.

        A block is considered a formula when more than the configured fraction
        of its non-whitespace characters belong to the configured font name.
        
        Nearby formula blocks are merged into clusters if their bounding boxes
        are within `formula_cluster_merge_tolerance` points of each other.
        """
        if not getattr(self.config, "save_formulas", True):
            return []

        threshold = float(getattr(self.config, "formula_font_threshold", 0.80))
        font_name = str(getattr(self.config, "formula_font_name", "Cambria Math"))
        
        # Use a specific tolerance for formula merging if available, 
        # otherwise fall back to cluster_merge_tolerance
        merge_tolerance = float(getattr(self.config, "formula_cluster_merge_tolerance", 20.0))

        try:
            text_dict = page.get_text("dict")
        except Exception as e:
            logger.error(f"Page {page_num}: Failed to get text dict: {e}")
            return []

        blocks = text_dict.get("blocks", [])
        if not blocks:
            return []

        # --- Step 1: Identify Candidate Formula Blocks ---
        candidates: List[Dict[str, Any]] = []
        
        for block_index, block in enumerate(blocks):
            if block.get("type") != 0:
                continue

            bbox = self._valid_bbox(block.get("bbox"))
            if bbox is None:
                continue

            if self._overlaps_excluded_bbox(bbox, exclude_bboxes):
                continue

            counts = self._font_counts(block, font_name)
            total_chars = counts["total_chars"]
            matching_chars = counts["cambria_math_chars"]

            if total_chars == 0:
                continue

            ratio = matching_chars / total_chars
            if ratio <= threshold:
                continue

            # Store candidate data for clustering
            candidates.append({
                "index": block_index,
                "bbox": bbox,
                "total_chars": total_chars,
                "matching_chars": matching_chars,
                "ratio": ratio,
            })

        if not candidates:
            return []

        # --- Step 2: Cluster Nearby Candidates ---
        clusters = self._merge_formula_blocks(candidates, merge_tolerance)

        # --- Step 3: Render Clusters ---
        results: List[FormulaInfo] = []

        for formula_num, cluster in enumerate(clusters, start=1):
            merged_bbox = cluster["bbox"]
            total_chars = cluster["total_chars"]
            matching_chars = cluster["matching_chars"]
            ratio = matching_chars / total_chars if total_chars > 0 else 0.0

            filename = self._make_filename(page_num, formula_num)
            info = FormulaInfo(
                formula_num=formula_num,   # now 1..N per page
                page_num=page_num,
                block_indices=cluster["indices"],
                bbox=merged_bbox,
                filename=filename,
                char_count=total_chars,
                cambria_math_char_count=matching_chars,
                cambria_math_ratio=ratio,
            )
            
            self._save_formula(page, info)
            
            if not info.saved:
                logger.warning(f"Page {page_num}, Cluster {formula_num}: Failed to save. Error: {info.save_error}")
            else:
                logger.info(f"Page {page_num}, Cluster {formula_num}: Saved as {info.filename} (merged {len(cluster['indices'])} blocks)")

            results.append(info)
            formula_num += 1

        return results

    def _merge_formula_blocks(
        self,
        candidates: List[Dict[str, Any]],
        tolerance: float,
    ) -> List[Dict[str, Any]]:
        """Merge candidate formula blocks that are spatially close.

        Uses a simple union-find or greedy clustering approach based on bbox distance.
        Two blocks are merged if the gap between their bboxes is less than `tolerance`.
        """
        if not candidates:
            return []

        # Sort candidates by Y position (top to bottom) to handle vertical flow first,
        # then X for horizontal alignment within similar Y ranges.
        # For simplicity, we use a greedy approach: 
        # 1. Sort by y0
        # 2. Iterate and merge if close in Y AND overlapping/close in X
        
        sorted_candidates = sorted(candidates, key=lambda c: (c["bbox"][1], c["bbox"][0]))
        
        clusters: List[Dict[str, Any]] = []
        used = set()

        for i, cand_a in enumerate(sorted_candidates):
            if i in used:
                continue
            
            # Start a new cluster with cand_a
            current_cluster_indices = [cand_a["index"]]
            current_bbox = list(cand_a["bbox"])  # mutable copy
            current_total_chars = cand_a["total_chars"]
            current_matching_chars = cand_a["matching_chars"]
            used.add(i)

            # Try to merge other candidates into this cluster
            changed = True
            while changed:
                changed = False
                for j, cand_b in enumerate(sorted_candidates):
                    if j in used:
                        continue
                    
                    # Check if cand_b is close to the current merged bbox
                    if self._is_close_bbox(current_bbox, cand_b["bbox"], tolerance):
                        # Merge cand_b into current cluster
                        used.add(j)
                        current_cluster_indices.append(cand_b["index"])
                        current_total_chars += cand_b["total_chars"]
                        current_matching_chars += cand_b["matching_chars"]
                        
                        # Expand bbox to include cand_b
                        current_bbox[0] = min(current_bbox[0], cand_b["bbox"][0])
                        current_bbox[1] = min(current_bbox[1], cand_b["bbox"][1])
                        current_bbox[2] = max(current_bbox[2], cand_b["bbox"][2])
                        current_bbox[3] = max(current_bbox[3], cand_b["bbox"][3])
                        
                        changed = True
            
            clusters.append({
                "indices": current_cluster_indices,
                "bbox": tuple(current_bbox),
                "total_chars": current_total_chars,
                "matching_chars": current_matching_chars,
            })

        return clusters

    @staticmethod
    def _is_close_bbox(
        bbox_a: Sequence[float],
        bbox_b: Sequence[float],
        tolerance: float,
    ) -> bool:
        """Check if two bboxes are within `tolerance` points of each other.

        This checks if the gap between them is <= tolerance in both X and Y dimensions.
        If they overlap, the gap is negative (0), so they are always close.
        """
        ax0, ay0, ax1, ay1 = bbox_a
        bx0, by0, bx1, by1 = bbox_b

        # Calculate horizontal gap
        if ax1 < bx0:  # A is left of B
            h_gap = bx0 - ax1
        elif bx1 < ax0:  # B is left of A
            h_gap = ax0 - bx1
        else:
            h_gap = 0.0  # Overlapping in X

        # Calculate vertical gap
        if ay1 < by0:  # A is above B
            v_gap = by0 - ay1
        elif by1 < ay0:  # B is above A
            v_gap = ay0 - by1
        else:
            v_gap = 0.0  # Overlapping in Y

        return h_gap <= tolerance and v_gap <= tolerance

    def count_formulas(
        self,
        page: pymupdf.Page,
        exclude_bboxes: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    ) -> int:
        """Return the number of formula-like text blocks without rendering them.
        
        Note: This currently counts individual blocks, not clusters. 
        If you need cluster count for pre-scanning, this might be slightly off 
        if merging happens, but it's usually acceptable for numbering offsets.
        """
        if not getattr(self.config, "save_formulas", True):
            return 0

        threshold = float(getattr(self.config, "formula_font_threshold", 0.80))
        font_name = str(getattr(self.config, "formula_font_name", "Cambria Math"))
        
        try:
            text_dict = page.get_text("dict")
        except Exception:
            return 0

        count = 0
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            bbox = self._valid_bbox(block.get("bbox"))
            if bbox is None or self._overlaps_excluded_bbox(bbox, exclude_bboxes):
                continue

            counts = self._font_counts(block, font_name)
            if counts["total_chars"] == 0:
                continue
            if counts["cambria_math_chars"] / counts["total_chars"] > threshold:
                count += 1

        return count

    def _save_formula(self, page: pymupdf.Page, info: FormulaInfo) -> None:
        self.media_dir.mkdir(parents=True, exist_ok=True)

        try:
            text_dict = page.get_text("rawdict")
            tight = None
            formula_rect = pymupdf.Rect(info.bbox)

            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        for char in span.get("chars", []):
                            char_rect = pymupdf.Rect(char["bbox"])

                            # Keep characters belonging to this formula cluster.
                            if not char_rect.intersects(formula_rect):
                                continue

                            tight = char_rect if tight is None else tight | char_rect

            if tight is None:
                tight = formula_rect

            # Optional small safety margin.
            padding = float(getattr(self.config, "formula_padding", 1.0))
            tight.x0 = max(0, tight.x0 - padding)
            tight.y0 = max(0, tight.y0 - padding)
            tight.x1 = min(page.rect.width, tight.x1 + padding)
            tight.y1 = min(page.rect.height, tight.y1 + padding)

            scale = float(getattr(self.config, "formula_render_scale", 2.0))
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(scale, scale),
                clip=tight,
                alpha=False,
            )
            pixmap.save(str(self.media_dir / info.filename))
            info.saved = True

        except Exception as exc:
            info.saved = False
            info.save_error = f"{type(exc).__name__}: {exc}"
            info.skipped_reason = "formula render/save failed"




    @staticmethod
    def _font_counts(
        block: Dict[str, Any],
        font_name: str = "Cambria Math",
    ) -> Dict[str, int]:
        total_chars = 0
        cambria_math_chars = 0

        target_normalized = font_name.lower().replace(" ", "")

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = str(span.get("text", ""))
                if not text:
                    continue

                char_count = sum(1 for char in text if not char.isspace())
                if char_count == 0:
                    continue

                total_chars += char_count
                
                actual_font = str(span.get("font", "")).lower().replace(" ", "")
                
                if target_normalized in actual_font:
                    cambria_math_chars += char_count

        return {
            "total_chars": total_chars,
            "cambria_math_chars": cambria_math_chars,
        }

    @staticmethod
    def _valid_bbox(value: Any) -> Optional[Tuple[float, float, float, float]]:
        if not value or len(value) < 4:
            return None
        try:
            x0, y0, x1, y1 = map(float, value[:4])
        except (TypeError, ValueError):
            return None
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    @staticmethod
    def _overlaps_excluded_bbox(
        bbox: Tuple[float, float, float, float],
        excluded: Optional[Sequence[Tuple[float, float, float, float]]],
    ) -> bool:
        if not excluded:
            return False

        x0, y0, x1, y1 = bbox
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area <= 0:
            return False

        for other in excluded:
            ox0, oy0, ox1, oy1 = other
            ix0 = max(x0, ox0)
            iy0 = max(y0, oy0)
            ix1 = min(x1, ox1)
            iy1 = min(y1, oy1)
            intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            if intersection >= 0.9 * min(
                area,
                max(0.0, ox1 - ox0) * max(0.0, oy1 - oy0),
            ):
                return True
        return False

    @staticmethod
    def _make_filename(page_num: int, formula_num: int) -> str:
        return f"formula_page_{page_num:04d}_number_{formula_num:04d}.png"
