"""Image extraction and persistence for PDF pages."""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pymupdf

from layout import BoundingBox
from overlap import check_margin_violation


@dataclass
class ImageInfo:
    """Information about an extracted image."""

    img_idx: int
    xref: int
    page_num: int
    saved: bool
    filename: Optional[str]
    filepath: Optional[str]
    skipped_reason: Optional[str]
    width: int = 0
    height: int = 0
    original_format: str = "unknown"
    bbox: Optional[BoundingBox] = None


class ImageExtractor:
    """Extract and save images from PDF pages."""

    # Images that are at least this large in both dimensions are considered
    # near-full-page images and are never saved.
    NEAR_FULL_PAGE_RATIO = 0.90

    # If text blocks completely covered by an image occupy at least this
    # fraction of the image area, the image is skipped.
    TEXT_COVERAGE_RATIO = 0.80


    # Small tolerance for PDF floating-point/layout differences when deciding
    # whether an image completely covers a text block. This prevents visually
    # coincident image/text boxes from being treated as unrelated objects just
    # because one edge differs by a point or two.
    TEXT_CONTAINMENT_PERCENT_TOLERANCE = 0.05

    def __init__(self, config, pdf_filename: str, image_dir: str):
        """Initialize extractor with configuration and output directory."""
        self.config = config
        self.pdf_filename = pdf_filename
        self.image_dir = image_dir

    def extract_images(
        self,
        page: pymupdf.Page,
        page_num: int,
        text_blocks_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> Tuple[List[ImageInfo], List[Tuple[float, float, float, float]]]:
        """Extract images from a page.

        Returns:
            A tuple of:
                - ``image_list``: list of :class:`ImageInfo` objects for all images found.
                - ``image_bboxes``: list of bounding boxes that were actually saved.
        """
        image_list: List[ImageInfo] = []
        image_bboxes: List[Tuple[float, float, float, float]] = []
        page_image_count = 0

        # Retrieve raw image metadata from PyMuPDF.
        image_info_list = page.get_image_info(xrefs=True)
        if not image_info_list:
            return image_list, image_bboxes

        for img_idx, img_info in enumerate(image_info_list):
            xref = img_info.get("xref")
            img_bbox = img_info.get("bbox")
            img_width = img_info.get("width", 0)
            img_height = img_info.get("height", 0)

            if not img_bbox or len(img_bbox) != 4:
                image_list.append(
                    ImageInfo(
                        img_idx=img_idx + 1,
                        xref=xref,
                        page_num=page_num,
                        saved=False,
                        filename=None,
                        filepath=None,
                        skipped_reason="missing or invalid image bbox",
                        width=img_width,
                        height=img_height,
                    )
                )
                continue

            # Process each image through validation, conflict resolution and optional saving.
            image_info = self._process_image(
                img_idx,
                xref,
                page_num,
                img_bbox,
                img_width,
                img_height,
                text_blocks_bboxes,
                page_image_count,
                page,
            )
            image_list.append(image_info)

            # If the image passed all checks and is to be saved, remember its bbox.
            if image_info.saved:
                image_bboxes.append(img_bbox)
                page_image_count += 1

        return image_list, image_bboxes

    def _process_image(
        self,
        img_idx: int,
        xref: int,
        page_num: int,
        img_bbox: Tuple[float, float, float, float],
        img_width: int,
        img_height: int,
        text_blocks_bboxes: Optional[List[Tuple[float, float, float, float]]],
        page_image_count: int,
        page: pymupdf.Page,
    ) -> ImageInfo:
        """Validate and possibly save a single image.

        Rules applied:

        1. Near-full-page images are never saved.
           An image is considered near-full-page when its width is at least
           ``NEAR_FULL_PAGE_RATIO`` of the page width and its height is at least
           ``NEAR_FULL_PAGE_RATIO`` of the page height.

        2. Very small images are skipped using ``config.min_size_px``.

        3. Images violating margin rules are skipped.

        4. For non-full-page images, if text blocks completely inside the image
           collectively cover at least ``TEXT_COVERAGE_RATIO`` of the image width
           and height, the image is skipped.

        5. The per-page maximum item limit is applied.

        6. Saving happens only when ``config.save_images`` is enabled.
        """
        bbox = BoundingBox.from_tuple(img_bbox)
        page_width = page.rect.width
        page_height = page.rect.height

        # ------------------------------------------------------------------
        # Rule 1: near-full-page images are never saved.
        # Text extraction is independent and remains available in the output.
        # ------------------------------------------------------------------
        if self._is_near_full_page(bbox, page_width, page_height):
            return ImageInfo(
                img_idx=img_idx + 1,
                xref=xref,
                page_num=page_num,
                saved=False,
                filename=None,
                filepath=None,
                skipped_reason=(
                    f"near-full-page image "
                    f"({bbox.width:.0f}x{bbox.height:.0f}pts "
                    f"vs page {page_width:.0f}x{page_height:.0f}pts)"
                ),
                width=img_width,
                height=img_height,
                bbox=bbox,
            )

        # ------------------------------------------------------------------
        # Rule 2: skip tiny images.
        # ------------------------------------------------------------------
        min_dimension = min(img_width, img_height)
        if min_dimension < self.config.min_size_px:
            return ImageInfo(
                img_idx=img_idx + 1,
                xref=xref,
                page_num=page_num,
                saved=False,
                filename=None,
                filepath=None,
                skipped_reason=f"below {self.config.min_size_px}px threshold",
                width=img_width,
                height=img_height,
                bbox=bbox,
            )

        # ------------------------------------------------------------------
        # Rule 3: margin violation.
        # ------------------------------------------------------------------
        if check_margin_violation(bbox, page_width, page_height, self.config):
            return ImageInfo(
                img_idx=img_idx + 1,
                xref=xref,
                page_num=page_num,
                saved=False,
                filename=None,
                filepath=None,
                skipped_reason="Content lies on outer margin",
                width=img_width,
                height=img_height,
                bbox=bbox,
            )

        # ------------------------------------------------------------------
        # Rule 4: skip image if contained text blocks dominate the image area.
        # ------------------------------------------------------------------
        if self._covered_text_fills_image(bbox, text_blocks_bboxes):
            return ImageInfo(
                img_idx=img_idx + 1,
                xref=xref,
                page_num=page_num,
                saved=False,
                filename=None,
                filepath=None,
                skipped_reason="image is mostly covered by contained text blocks",
                width=img_width,
                height=img_height,
                bbox=bbox,
            )



        # ------------------------------------------------------------------
        # Rule 5: max items per page.
        # ------------------------------------------------------------------
        if (
            self.config.max_items_per_page > 0
            and page_image_count >= self.config.max_items_per_page
        ):
            return ImageInfo(
                img_idx=img_idx + 1,
                xref=xref,
                page_num=page_num,
                saved=False,
                filename=None,
                filepath=None,
                skipped_reason="max limit reached",
                width=img_width,
                height=img_height,
                bbox=bbox,
            )

        # ------------------------------------------------------------------
        # Rule 6: final save decision.
        # ------------------------------------------------------------------
        if self.config.save_images:
            return ImageInfo(
                img_idx=img_idx + 1,
                xref=xref,
                page_num=page_num,
                saved=True,
                filename=f"{self.pdf_filename}_page_{page_num:04d}_img_{img_idx + 1:02d}.png",
                filepath=None,
                skipped_reason=None,
                width=img_width,
                height=img_height,
                original_format="png",
                bbox=bbox,
            )

        return ImageInfo(
            img_idx=img_idx + 1,
            xref=xref,
            page_num=page_num,
            saved=False,
            filename=None,
            filepath=None,
            skipped_reason="save_images=False",
            width=img_width,
            height=img_height,
            bbox=bbox,
        )

    def save_all(self, images: List[ImageInfo], doc: pymupdf.Document) -> None:
        """Write all successfully validated images to disk."""
        if any(img.saved for img in images):
            os.makedirs(self.image_dir, exist_ok=True)

        for img in images:
            if img.saved and img.filename is not None:
                img.filepath = os.path.join(self.image_dir, img.filename)
                self._save_image_to_disk(img, doc)

    def _save_image_to_disk(self, img: ImageInfo, doc: pymupdf.Document) -> None:
        """Perform the actual file I/O for a saved image."""
        try:
            pix = pymupdf.Pixmap(doc, img.xref)
            png_data = pix.tobytes("png")

            if img.filepath is None:
                return

            with open(img.filepath, "wb") as f:
                f.write(png_data)

            print(f"Saved image: {img.filename} ({img.width}x{img.height}px)")
        except Exception as exc:
            img.saved = False
            img.skipped_reason = f"Save failed: {str(exc)}"

    def _is_near_full_page(
        self,
        bbox: BoundingBox,
        page_width: float,
        page_height: float,
    ) -> bool:
        """Return True when the image occupies at least 90% of both page dimensions."""
        if page_width <= 0 or page_height <= 0:
            return False

        width_ratio = bbox.width / page_width
        height_ratio = bbox.height / page_height

        return (
            width_ratio >= self.NEAR_FULL_PAGE_RATIO
            and height_ratio >= self.NEAR_FULL_PAGE_RATIO
        )


    def _covered_text_fills_image(
        self,
        image_bbox: BoundingBox,
        text_blocks_bboxes: Optional[List[Tuple[float, float, float, float]]],
    ) -> bool:
        """
        Return True when contained text blocks cover at least TEXT_COVERAGE_RATIO 
        of the image area.
        
        Text blocks are considered "contained" if they lie within the image bounds, 
        allowing for a percentage-based tolerance to handle slight layout shifts 
        or blocks that are slightly larger than the image (e.g., up to 105%).
        """
        if not text_blocks_bboxes:
            return False

        image_width = image_bbox.width
        image_height = image_bbox.height
        if image_width <= 0 or image_height <= 0:
            return False

        # Calculate absolute tolerance based on image dimensions and percentage
        x_tolerance = self.TEXT_CONTAINMENT_PERCENT_TOLERANCE * image_width
        y_tolerance = self.TEXT_CONTAINMENT_PERCENT_TOLERANCE * image_height
        
        covered_rects: List[Tuple[float, float, float, float]] = []

        for text_bbox_tuple in text_blocks_bboxes:
            if not text_bbox_tuple or len(text_bbox_tuple) != 4:
                continue

            text_bbox = BoundingBox.from_tuple(text_bbox_tuple)
            
            # Skip degenerate boxes
            if text_bbox.width <= 0 or text_bbox.height <= 0:
                continue

            # Check spatial containment with percentage-based tolerance.
            # This allows text blocks to be slightly larger than the image 
            # (up to TEXT_CONTAINMENT_PERCENT_TOLERANCE on each side) and still be counted.
            is_contained = (
                text_bbox.x0 >= image_bbox.x0 - x_tolerance
                and text_bbox.y0 >= image_bbox.y0 - y_tolerance
                and text_bbox.x1 <= image_bbox.x1 + x_tolerance
                and text_bbox.y1 <= image_bbox.y1 + y_tolerance
            )
            
            if not is_contained:
                continue

            # Clip the text block to the image bounds for accurate area calculation.
            # This ensures we only count the overlap, not the part sticking out.
            clipped_x0 = max(image_bbox.x0, text_bbox.x0)
            clipped_y0 = max(image_bbox.y0, text_bbox.y0)
            clipped_x1 = min(image_bbox.x1, text_bbox.x1)
            clipped_y1 = min(image_bbox.y1, text_bbox.y1)

            # Only add if there is actual overlap
            if clipped_x1 > clipped_x0 and clipped_y1 > clipped_y0:
                covered_rects.append((clipped_x0, clipped_y0, clipped_x1, clipped_y1))

        if not covered_rects:
            return False

        # Calculate the union area of all contained text blocks to handle overlaps correctly
        covered_area = self._rectangle_union_area(covered_rects)
        image_area = image_width * image_height
        
        if image_area <= 0:
            return False

        coverage_ratio = covered_area / image_area

        return coverage_ratio >= self.TEXT_COVERAGE_RATIO


    @staticmethod
    def _rectangle_union_area(
        rectangles: List[Tuple[float, float, float, float]],
    ) -> float:
        """Return the union area of axis-aligned rectangles."""
        if not rectangles:
            return 0.0

        x_edges = sorted(
            {x for x0, _, x1, _ in rectangles for x in (x0, x1)}
        )
        union_area = 0.0

        for x0, x1 in zip(x_edges, x_edges[1:]):
            strip_width = x1 - x0
            if strip_width <= 0:
                continue

            y_intervals = [
                (y0, y1)
                for rx0, y0, rx1, y1 in rectangles
                if rx0 < x1 and rx1 > x0 and y1 > y0
            ]
            if not y_intervals:
                continue

            y_intervals.sort()
            covered_y = 0.0
            current_y0, current_y1 = y_intervals[0]

            for y0, y1 in y_intervals[1:]:
                if y0 <= current_y1:
                    current_y1 = max(current_y1, y1)
                else:
                    covered_y += current_y1 - current_y0
                    current_y0, current_y1 = y0, y1

            covered_y += current_y1 - current_y0
            union_area += strip_width * covered_y

        return union_area
