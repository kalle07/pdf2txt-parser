"""Main PDF conversion pipeline and multiprocessing orchestration."""

# ------------------------------------------------------------
# Standard‑library imports
# ------------------------------------------------------------
import json                # JSON encoder/decoder – used for table JSON output
import os                  # OS‑level utilities (path handling, file I/O)
import time                # Time‑stamping / elapsed‑time calculations
import logging
import traceback 
from concurrent.futures import ProcessPoolExecutor, as_completed
#   –‑ ProcessPoolExecutor runs the worker pool; as_completed yields futures
from dataclasses import dataclass   # Makes ConversionConfig & ConversionStats dataclasses
from enum import Enum               # Defines ConversionMode enum
from pathlib import Path            # Path objects for filesystem paths
from typing import (Any, Callable, Dict, List, Optional, Tuple)  # Generic type hints


# ------------------------------------------------------------
# Third‑party imports (the only external dependencies)
# ------------------------------------------------------------
import psutil                 # Provides cpu_count(logical=False) → physical cores
import pymupdf                # PyMuPDF – PDF parsing, page access, drawing/image extraction

# ------------------------------------------------------------
# Local package imports – re‑exported symbols are pulled in
#    later by the package’s __init__.py so users can write
#    `from pdf_parser import PDFConverter` etc.
# ------------------------------------------------------------
from drawings import DrawingExtractor, DrawingInfo
from images import ImageExtractor, ImageInfo
from formula import FormulaExtractor, FormulaInfo
from layout import BoundingBox, passes_size_filter
from overlap import (
    calculate_bbox_overlap,
    calculate_bbox_distance,
    check_margin_violation,
)
from post_process import count_characters, count_words, fix_hyphenated_lines
from tables import TableData, TableProcessor
#   –‑ Table cleaning, structuring and JSON‑serialization helpers


# ----------------------------------------------------------------------
# Enumerations and configuration dataclasses
# ----------------------------------------------------------------------
class ConversionMode(Enum):
    """Supported conversion modes (currently only `full_conversion` is used)."""
    FULL_CONVERSION = "full_conversion"
    POST_PROCESS_ONLY = "post_process_only"


@dataclass
class ConversionConfig:
    """Centralized configuration for PDF conversion.

    All conversion‑related thresholds and flags are gathered here so they can be
    easily shared between processes via a plain ``dict``.
    """

    save_images: bool = True
    save_drawings: bool = True
    min_size_px: int = 100
    max_items_per_page: int = 10

    row_tolerance: float = 0.0195   # fraction of page height (1.95 %)
    col_tolerance: float = 0.016    # fraction of page width  (1.6 %)
    cluster_merge_tolerance: float = 20.0
    bbox_padding: float = 20.0
    min_block_size_px: int = 5

    min_items_per_cluster: int = 8
    include_small_text_blocks: bool = True

    enable_margin_check: bool = True
    margin_left: float = 0.05
    margin_right: float = 0.95
    margin_top: float = 0.06
    margin_bottom: float = 0.94

    small_text_max_chars: int = 50
    hyphen_fix_enabled: bool = True
    include_metadata: bool = True
    save_formulas: bool = True
    formula_font_name: str = "Cambria Math"
    formula_font_threshold: float = 0.50
    formula_render_scale: float = 2.0
    formula_padding: float = 5.0
    formula_cluster_merge_tolerance: float = 5.0


@dataclass
class ConversionStats:
    """Accumulated statistics for a single page conversion.

    The ``to_dict()`` method filters out zero‑valued counters, making it easy
    to serialize only the meaningful metrics.
    """

    pages_processed: int = 0
    images_saved: int = 0
    images_skipped: int = 0
    drawings_saved: int = 0
    drawings_skipped_size: int = 0
    drawings_skipped_overlap: int = 0
    drawings_skipped_min_items: int = 0
    drawings_skipped_margin: int = 0
    drawings_skipped_limit: int = 0
    drawings_error: int = 0
    tables_skipped: int = 0
    tables_skipped_margin: int = 0
    tables_skipped_limit: int = 0
    hyphens_fixed: int = 0
    table_hyphens_fixed: int = 0
    word_count: int = 0
    char_count: int = 0
    formulas_saved: int = 0

    def to_dict(self) -> Dict[str, int]:
        """Return only the counters that are greater than zero."""
        return {k: v for k, v in self.__dict__.items() if v > 0}


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def get_physical_cores() -> int:
    """Return the number of **physical** CPU cores (ignoring logical hyper‑threads).

    ``psutil.cpu_count(logical=False)`` may return ``None`` on some platforms,
    so we default to ``1`` to avoid division‑by‑zero later.
    """
    count = psutil.cpu_count(logical=False)
    return max(1, count if count else 1)


def _clean_pdf_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return JSON-friendly PDF metadata with empty values removed."""
    if not metadata:
        return {}
    return {
        str(key): value
        for key, value in metadata.items()
        if value not in (None, "")
    }


def _process_pages_worker(
    pdf_path: str,
    page_range: Tuple[int, int],
    total_pages: int,
    config_dict: Dict[str, Any],
    media_dir: str,
    pdf_filename: str,
) -> List[Tuple[int, str, Dict[str, int]]]:
    """Worker function executed in a separate process for a chunk of pages.

    It loads the PDF once, processes all pages in the chunk, then closes it.
    Returns a list of (page_num, content, stats) tuples — one per page in the chunk.
    Exceptions are handled per-page to ensure a single bad page doesn't kill the whole chunk.
    """
    config = ConversionConfig(**config_dict)
    results: List[Tuple[int, str, Dict[str, int]]] = []

    try:
        doc = pymupdf.Document(pdf_path)
        for page_num in range(page_range[0], page_range[1] + 1):
            page_stats = ConversionStats()
            
            try:
                page = doc[page_num - 1]
                processor = PageProcessor(
                    config=config,
                    pdf_filename=pdf_filename,
                    media_dir=media_dir,
                    stats=page_stats,
                    doc=doc,
                )
                page_content = processor.process(page, page_num, total_pages)
                results.append((page_num, page_content, page_stats.to_dict()))
            except Exception as page_exc:
                # Log the error for this specific page and continue to the next

                logging.error(f"Error processing page {page_num} of {pdf_filename}: {page_exc}")
                
                # Add an error entry for this specific page
                error_content = f"\n\nERROR ON PAGE {page_num}: {str(page_exc)}\n\n"
                error_stats = {
                    "pages_processed": 0,
                    "pages_failed": 1,
                    "error": str(page_exc),
                }
                results.append((page_num, error_content, error_stats))
        
        doc.close()
        return results
    except Exception as exc:
        # This block only catches errors that happen before the loop starts
        # (e.g., file open failure) or if the loop itself crashes unexpectedly.
        # In that case, we return an error for the first page as a fallback.
        chunk_size = page_range[1] - page_range[0] + 1
        return [
            (
                page_range[0],
                f"\n\nERROR ON PAGE {page_range[0]}: {str(exc)}\n\n",
                {
                    "pages_processed": 0,
                    "pages_attempted": chunk_size,
                    "pages_failed": chunk_size,
                    "error": str(exc),
                },
            )
        ]



def create_processing_tasks(
    pdf_files: List[Dict[str, Any]], num_cores: int
) -> List[Dict[str, Any]]:
    """Create a list of processing tasks based on file size and core count.

    * Small files (< 32 pages) get a single core each.
    * Large files are split into roughly equal chunks, each processed by a
      separate core to maximise parallelism.
    The returned ``tasks`` list is later fed to ``ProcessPoolExecutor``.
    """
    tasks: List[Dict[str, Any]] = []

    for file_info in pdf_files:
        page_count = file_info["page_count"]

        if page_count < 32:
            # Small file — process all pages in a single task.
            tasks.append(
                {
                    "type": "single_file",
                    "pdf_path": file_info["path"],
                    "page_count": page_count,
                    "pdf_filename": file_info["filename"],
                    "cores_used": 1,
                    "page_range": (1, page_count),
                }
            )
        else:
            # Large file — split into 8-page chunks so workers stay
            # responsive and the executor can interleave files.
            chunk_size = 8
            current_page = 1
            while current_page <= page_count:
                pages_in_chunk = min(chunk_size, page_count - current_page + 1)
                tasks.append(
                    {
                        "type": "split_file",
                        "pdf_path": file_info["path"],
                        "total_page_count": page_count,
                        "pdf_filename": file_info["filename"],
                        "cores_used": 1,
                        "page_range": (current_page, current_page + pages_in_chunk - 1),
                    }
                )
                current_page += pages_in_chunk

    return tasks


def assemble_results(
    results: List[Tuple[int, str, Dict[str, int]]],
    pdf_filename: str,
    output_file: str,
    media_dir: str,
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[ConversionConfig] = None,
) -> Dict[str, Any]:
    """Collect page results, write them to the output file, and return aggregated stats.

    The function:

    1. Sorts results by page number to preserve document order.
    2. Extracts per‑page statistics and aggregates them into a global dict.
    3. Writes a human‑readable header (PDF name, media folder, config status) followed by the
       page contents.
    The aggregated stats are returned for later merging.
    """
    sorted_results = sorted(results, key=lambda x: x[0])

    total_stats = {
        "pages_processed": 0,
        "images_saved": 0,
        "drawings_saved": 0,
        "hyphens_fixed": 0,
        "table_hyphens_fixed": 0,
        "tables_skipped": 0,
        "word_count": 0,
        "char_count": 0,
        "formulas_saved": 0,
        "errors": 0,
    }

    for _, _, page_stats in sorted_results:
        if "error" in page_stats:
            total_stats["errors"] += page_stats.get("pages_failed", 1)
        else:
            total_stats["pages_processed"] += page_stats.get("pages_processed", 1)
            total_stats["images_saved"] += page_stats.get("images_saved", 0)
            total_stats["drawings_saved"] += page_stats.get("drawings_saved", 0)
            total_stats["hyphens_fixed"] += page_stats.get("hyphens_fixed", 0)
            total_stats["table_hyphens_fixed"] += page_stats.get("table_hyphens_fixed", 0)
            total_stats["tables_skipped"] += page_stats.get("tables_skipped", 0)
            total_stats["word_count"] += page_stats.get("word_count", 0)
            total_stats["char_count"] += page_stats.get("char_count", 0)
            total_stats["formulas_saved"] += page_stats.get("formulas_saved", 0)

    with open(output_file, "w", encoding="utf-8") as output:
        output.write(f"PDF: {pdf_filename}\n")
        output.write(f"Media Directory: {os.path.basename(media_dir)}\n")
        
        # --- Configuration Status ---
        if config is not None:
            output.write(f"Hyphen Fix: {'enabled' if config.hyphen_fix_enabled else 'disabled'}\n")
            output.write(f"Image Extraction: {'enabled' if config.save_images else 'disabled'}\n")
            output.write(f"Drawing Extraction: {'enabled' if config.save_drawings else 'disabled'}\n")
            output.write(f"Formula Extraction: {'enabled' if config.save_formulas else 'disabled'}\n")
        # ---------------------------------------

        output.write(f"Word Count: {total_stats['word_count']}\n")
        output.write(f"Character Count: {total_stats['char_count']}\n")
        
        # --- Conditional counts: only shown when the feature is enabled ---
        if config is None or config.save_formulas:
            output.write(f"Formulas Saved: {total_stats['formulas_saved']}\n")
        if config is None or config.save_images:
            output.write(f"Images Saved: {total_stats['images_saved']}\n")
        if config is None or config.save_drawings:
            output.write(f"Drawings Saved: {total_stats['drawings_saved']}\n")            
        if config is None or config.hyphen_fix_enabled:
            output.write(f"Text Hyphens Fixed: {total_stats['hyphens_fixed']}\n")
            output.write(f"Table Hyphens Fixed: {total_stats['table_hyphens_fixed']}\n")
        # ------------------------------------------------------------------
        
        output.write("=" * 60 + "\n\n")

        cleaned_metadata = _clean_pdf_metadata(metadata)
        if config is None or config.include_metadata:
            if cleaned_metadata:
                output.write("PDF METADATA\n\n")
                output.write("```json\n")
                output.write(json.dumps(cleaned_metadata, indent=2, ensure_ascii=False))
                output.write("\n```\n\n")

        for _, content, _ in sorted_results:
            output.write(content)

    return total_stats


# ----------------------------------------------------------------------
# PDF validation
# ----------------------------------------------------------------------
class PDFValidator:
    """Validate PDF files before they are processed."""

    @staticmethod
    def validate_file(pdf_path: str) -> Dict[str, Any]:
        """Inspect a single PDF file for validity, corruption, and protection.

        The returned dict contains:

        * ``filename`` – original file name,
        * ``path`` – absolute path,
        * ``valid`` – ``True`` if the file can be opened,
        * ``page_count`` – number of pages (``None`` if invalid),
        * ``reason`` – human‑readable explanation when ``valid`` is ``False``,
        * ``is_repaired`` – ``True`` if PyMuPDF repaired the file,
        * ``is_protected`` – ``True`` if the file is encrypted.
        """
        filename = Path(pdf_path).name
        result = {
            "filename": filename,
            "path": str(pdf_path),
            "valid": False,
            "page_count": None,
            "reason": None,
            "is_repaired": False,
            "is_protected": False,
            "metadata": {},
        }

        try:
            pymupdf.TOOLS.reset_mupdf_warnings()
            doc = pymupdf.open(pdf_path)

            if doc.is_encrypted:
                result["is_protected"] = True
                result["valid"] = False
                result["reason"] = "File is encrypted/protected"
                doc.close()
                return result

            if doc.is_repaired:
                result["is_repaired"] = True

            warnings = pymupdf.TOOLS.mupdf_warnings(reset=True)
            result["valid"] = True
            result["page_count"] = len(doc)
            result["metadata"] = _clean_pdf_metadata(doc.metadata)
            result["metadata"]["page_count"] = len(doc)

            if warnings:
                print(f"Warning for '{filename}': {warnings}")

            doc.close()
        except pymupdf.FileDataError as exc:
            result["reason"] = f"Corrupted file or invalid PDF structure: {exc}"
        except Exception as exc:
            result["reason"] = f"Open failed: {type(exc).__name__}: {exc}"

        return result


# ----------------------------------------------------------------------
# Page processing
# ----------------------------------------------------------------------
class PageProcessor:
    """Process a single PDF page and extract text, tables, images, and drawings."""

    def __init__(
        self,
        config: ConversionConfig,
        pdf_filename: str,
        media_dir: str,
        stats: ConversionStats,
        doc: Optional[pymupdf.Document] = None,
    ):
        self.config = config
        self.pdf_filename = pdf_filename
        self.media_dir = media_dir
        self.stats = stats
        self.doc = doc

    def process(
        self,
        page: pymupdf.Page,
        page_num: int,
        total_pages: int,
        formula_start_num: int = 1,
    ) -> str:
        """Extract and format all content from a single page.

        The method orchestrates extraction of tables, text blocks, images,
        and drawings, then assembles them into a single string that is written
        to the final ``.txt`` output.
        """
        page_content: List[str] = []
        page_content.append(f"\n\nPage Number: {page_num} of {total_pages}\n\n\n")

        # Containers for bounding boxes that help avoid duplicate extractions
        reference_bboxes: Dict[str, List[Tuple[float, float, float, float]]] = {
            "tables": [],
            "text_blocks": [],
            "formula_blocks": [],
            "images": [],
            "attached_text_blocks": [],
            "drawing_blocks": [],
        }

        # ------------------------------------------------------------------
        # 1️⃣ Extract tables
        # ------------------------------------------------------------------
        page_content.extend(self._process_tables(page, page_num, reference_bboxes))

        # ------------------------------------------------------------------
        # 2️⃣ Extract Cambria Math-heavy formula blocks before normal text
        #    Formula blocks are treated as visual media and are therefore
        #    excluded from normal text extraction.
        # ------------------------------------------------------------------
        formula_extractor = None
        formula_list: List[FormulaInfo] = []
        if self.config.save_formulas:
            formula_extractor = FormulaExtractor(
                self.config,
                self.pdf_filename,
                self.media_dir,
            )
            formula_list = formula_extractor.extract_formulas(page, page_num)
            
            reference_bboxes["formula_blocks"] = [
                formula.bbox for formula in formula_list
                if formula.bbox is not None and isinstance(formula.bbox, (list, tuple)) and len(formula.bbox) == 4
            ]

        # ------------------------------------------------------------------
        # 3️⃣ Collect plain text blocks (excluding tables and formula blocks).
        #    Output is deliberately deferred until drawings are known, so any
        #    plain block covered by a drawing can be removed before emission.
        # ------------------------------------------------------------------
        plain_text_blocks = self._process_text_blocks(
            page, page_num, reference_bboxes
        )

        # ------------------------------------------------------------------
        # 4️⃣ Extract images (if enabled)
        # ------------------------------------------------------------------
        image_extractor = None
        images_on_page: List[ImageInfo] = []
        if self.config.save_images:
            image_extractor = ImageExtractor(
                self.config, self.pdf_filename, self.media_dir
            )
            images_on_page, image_bboxes = image_extractor.extract_images(
                page,
                page_num,
                reference_bboxes.get("text_blocks"),
            )
            reference_bboxes["images"] = image_bboxes

        # ------------------------------------------------------------------
        # 5️⃣ Extract drawings
        # ------------------------------------------------------------------
        drawing_extractor = DrawingExtractor(
            self.config, self.pdf_filename, self.media_dir
        )
        drawing_list, _, _ = drawing_extractor.extract_drawings(
            page,
            page_num,
            reference_bboxes,
            0,
            self.stats,
        )

        # ------------------------------------------------------------------
        # 6️⃣ Resolve conflicts between images and drawings that heavily overlap
        # ------------------------------------------------------------------
        self._resolve_image_drawing_conflicts(images_on_page, drawing_list)

        # ------------------------------------------------------------------
        # 6b️⃣ When drawings are not saved, treat each saved drawing as ONE
        #     independent text block.  Reserve its full drawing area first,
        #     remove overlapping plain-text blocks, then emit drawing text
        #     after all remaining plain text.
        # ------------------------------------------------------------------
        drawing_text_blocks: List[Tuple[Tuple[float, float, float, float], str, int]] = []
        
        # Only process drawings that are marked as saved (valid)
        valid_drawing_bboxes = [
            drawing_info.bbox.to_tuple()
            for drawing_info in drawing_list
            if drawing_info.saved and drawing_info.bbox is not None
        ]

        if valid_drawing_bboxes:
            reference_bboxes["drawing_blocks"] = valid_drawing_bboxes

            # Extract text from each valid drawing area
            drawing_text_blocks = self._extract_drawing_text_blocks(
                page, drawing_list
            )

            # Remove any plain text block that overlaps with a drawing area
            plain_text_blocks = [
                block
                for block in plain_text_blocks
                if not self._text_block_overlaps_drawing(
                    block[0], valid_drawing_bboxes
                )
            ]

        # ------------------------------------------------------------------
        # 7️⃣ Emit text in the required order:
        #     1) remaining plain text blocks
        #     2) drawing-contained text blocks
        # ------------------------------------------------------------------
        for _, fixed_text, fix_count in plain_text_blocks:
            self.stats.hyphens_fixed += fix_count
            self._record_text_counts(fixed_text)
            page_content.append(fixed_text + "\n\n")

        for _, fixed_text, fix_count in drawing_text_blocks:
            self.stats.hyphens_fixed += fix_count
            self._record_text_counts(fixed_text)
            # Optionally add a marker to indicate this text came from a drawing
            page_content.append("\n\n[Text from Drawing]\n")
            page_content.append(fixed_text + "\n\n")

        # ------------------------------------------------------------------
        # 8️⃣ Save extracted media to disk (only when enabled)
        # ------------------------------------------------------------------
        if image_extractor:
            image_extractor.save_all(images_on_page, self.doc)
        if drawing_extractor and self.config.save_drawings:
            drawing_extractor.save_all(drawing_list, page)

        # ------------------------------------------------------------------
        # 8️⃣ Append JSON metadata for formulas, images and drawings
        # ------------------------------------------------------------------
        page_content.extend(self._write_formula_info(formula_list, page_num))
        page_content.extend(self._write_image_info(images_on_page, page_num))
        page_content.extend(self._write_drawing_info(drawing_list, page_num))

        # Update global counter of processed pages
        self.stats.pages_processed += 1
        return "".join(page_content)

    # ----------------------------------------------------------------------
    # Helper methods used by ``process`` – each is documented inline
    # ----------------------------------------------------------------------
    def _resolve_image_drawing_conflicts(
        self,
        images_on_page: List[ImageInfo],
        drawing_list: List[DrawingInfo],
    ) -> None:
        """When an image and a drawing overlap heavily, keep only the larger one.

        Overlap ratio is computed as ``intersection_area / min(area_image, area_drawing)``.
        If the ratio exceeds 80 % the smaller item is marked as ``saved=False``.
        This prevents duplicate visual artefacts in the output.
        """
        if not (self.config.save_images and self.config.save_drawings):
            return

        for drawing_info in drawing_list:
            if not drawing_info.saved:
                continue
            for image_info in images_on_page:
                if not image_info.saved:
                    continue

                i_bbox = image_info.bbox
                d_box = drawing_info.bbox
                if i_bbox is None or d_box is None:
                    continue

                _, inter_area, has_overlap = calculate_bbox_overlap(d_box, i_bbox)
                if not has_overlap:
                    continue

                min_area = min(i_bbox.area, d_box.area)
                overlap_pct = inter_area / min_area if min_area > 0 else 0
                if overlap_pct <= 0.8:
                    continue

                # Keep the larger object, mark the smaller as skipped
                if i_bbox.area < d_box.area:
                    image_info.saved = False
                    image_info.skipped_reason = (
                        f"Overlap with larger drawing ({overlap_pct:.1%} coverage)"
                    )
                elif i_bbox.area > d_box.area:
                    drawing_info.saved = False
                    drawing_info.skipped_reason = (
                        f"Overlap with larger image ({overlap_pct:.1%} coverage)"
                    )

    def _process_tables(
        self,
        page: pymupdf.Page,
        page_num: int,
        reference_bboxes: Dict[str, List[Tuple[float, float, float, float]]],
    ) -> List[str]:
        """Extract all tables from the page, filter them, and convert to JSON.

        Tables that violate margin rules, exceed the per‑page item limit, or are
        deemed non‑useful are skipped.  Each retained table is formatted as a
        JSON block and appended to ``content``.
        """
        content: List[str] = []
        # Fast pre-check: the default "lines" strategy can only build a table
        # from vector lines or rectangles. Pages without any are skipped so we
        # avoid find_tables()'s expensive per-character bbox extraction — this
        # is the single largest cost in the pipeline for text-only pages.
        if not self._page_has_table_lines(page):
            return content
        tables_obj = page.find_tables()
        if not tables_obj or not tables_obj.tables:
            return content

        page_text_blocks = page.get_text("blocks")
        
        table_num = 0
        for table in tables_obj.tables:
            table_num += 1

            # Keep the parser output untouched. TableProcessor owns all
            # reconstruction/cleanup logic; this dict is only the raw source.
            raw_data = table.extract()
            raw_table = TableProcessor.make_raw_table(
                table,
                raw_data,
                page_num,
                table_num,
            )

            # Build the final logical matrix from bbox geometry, then clean it.
            cleaned_data, table_fix_count = TableProcessor.prepare_table_data(
                raw_table,
                hyphen_fix_enabled=self.config.hyphen_fix_enabled,
                return_fix_count=True,
            )
            table_bbox = BoundingBox.from_tuple(table.bbox).to_tuple()

            # ---- Margin check -------------------------------------------------
            if check_margin_violation(
                BoundingBox.from_tuple(table_bbox),
                page.rect.width,
                page.rect.height,
                self.config,
            ):
                self.stats.tables_skipped_margin += 1
                continue

            # ---- Max‑items‑per‑page limit -------------------------------------
            if (
                self.config.max_items_per_page > 0
                and len(reference_bboxes["tables"])
                >= self.config.max_items_per_page
            ):
                self.stats.tables_skipped_limit += 1
                continue

            # ---- Usefulness filter -------------------------------------------
            if not TableProcessor.is_useful_table(cleaned_data):
                self.stats.tables_skipped += 1
                continue

            # ---- Process and serialize ----------------------------------------
            table_data = TableProcessor.process_table(
                cleaned_data,
                page_num,
                table_num,
            )

            # Skip tables with no actual data rows (e.g., only headline/description)
            if table_data.data_rows_count == 0:
                self.stats.tables_skipped += 1
                continue

            self.stats.table_hyphens_fixed += table_fix_count

            # These blocks are removed from normal text output later.
            attached_bboxes = set(reference_bboxes.get("attached_text_blocks", []))
            
            table_data = TableProcessor.attach_nearby_text_blocks(
                table_data,
                BoundingBox.from_tuple(table_bbox),
                page_text_blocks,
                attached_bboxes,
                max_lines=2,
                max_chars=250,
                max_distance_px=10.0,
                max_overlap_px=2.0,
            )
            reference_bboxes["attached_text_blocks"] = list(attached_bboxes)

            self._record_table_counts(table_data)
            reference_bboxes["tables"].append(table_bbox)
            content.extend(self._format_table_output(table_data, page_num, table_num))

        return content

    @staticmethod
    def _page_has_table_lines(page: pymupdf.Page) -> bool:
        """Return True if the page contains any vector line or rectangle.

        The default ``find_tables`` strategy ("lines") derives table cells
        exclusively from vector lines and rectangle edges. A page with none
        cannot yield a table, so skipping ``find_tables`` there changes no
        output while avoiding its costly character-quad extraction.
        """
        for drawing in page.get_drawings():
            for item in drawing.get("items", []):
                if item and item[0] in ("l", "re"):
                    return True
        return False

    def _format_table_output(
        self, table_data: TableData, page_num: int, table_num: int
    ) -> List[str]:
        """Create a human‑readable section for a single table.

        The output includes a brief summary, the headline (if any), and the full
        table data serialized as pretty‑printed JSON within a fenced code block.
        """
        content = []
        content.append(f"\n\nTABLE {table_num} ON PAGE {page_num}\n\n")
        content.append(f"Type: {table_data.type}\n")
        content.append(f"Has Header: {table_data.has_header_row}\n")
        content.append(f"Data Rows: {table_data.data_rows_count}\n")

        table_json = {
            "table_number": table_num,
            "page_number": page_num,
            "headline": table_data.headline,
            "description": table_data.description,
            "data_rows": table_data.data_rows,
        }
        content.append("```json\n")
        content.append(json.dumps(table_json, indent=2, ensure_ascii=False))
        content.append("\n```\n\n")
        return content

    def _process_text_blocks(
        self,
        page: pymupdf.Page,
        page_num: int,
        reference_bboxes: Dict[str, List[Tuple[float, float, float, float]]],
    ) -> List[Tuple[Tuple[float, float, float, float], str, int]]:
        """Collect normal text blocks without emitting them yet.

        The returned records contain ``(bbox, normalized_text, hyphen_fix_count)``.
        Emission is deferred until drawings have been resolved so drawing-area
        overlap can suppress duplicate plain-text output.
        """
        content: List[Tuple[Tuple[float, float, float, float], str, int]] = []
        blocks = page.get_text("blocks")
        sorted_blocks = self._sort_blocks_by_layout(blocks, page)

        for block in sorted_blocks:
            if not block or len(block) < 5:
                continue

            x0, y0, x1, y1, text = block[:5]
            block_bbox = BoundingBox.from_tuple((x0, y0, x1, y1))
            block_bbox_tuple = (x0, y0, x1, y1)

            # Skip text blocks already attached to a table as headline/description.
            attached_bboxes = set(reference_bboxes.get("attached_text_blocks", []))
            if block_bbox_tuple in attached_bboxes:
                continue

            # ---- Margin filter ------------------------------------------------
            if check_margin_violation(
                block_bbox, page.rect.width, page.rect.height, self.config
            ):
                continue
            if not text.strip():
                continue

            # TABLE PRIORITY CHECKS (Tables take precedence over text blocks)
            should_skip = False

            for tx0, ty0, tx1, ty1 in reference_bboxes["tables"]:
                table_bbox = BoundingBox.from_tuple((tx0, ty0, tx1, ty1))

                # CHECK 1: Text block completely contained within table area
                if (
                    block_bbox.x0 >= table_bbox.x0
                    and block_bbox.y0 >= table_bbox.y0
                    and block_bbox.x1 <= table_bbox.x1
                    and block_bbox.y1 <= table_bbox.y1
                ):
                    should_skip = True
                    break

                # CHECK 2: High overlap with similar size
                overlap_pct, _, has_overlap = calculate_bbox_overlap(
                    block_bbox, table_bbox
                )

                if has_overlap and overlap_pct > 0.9:
                    if max(table_bbox.area, block_bbox.area) > 0:
                        size_diff = abs(
                            block_bbox.area - table_bbox.area
                        ) / max(table_bbox.area, block_bbox.area)
                        if size_diff <= 0.2:
                            should_skip = True
                            break

            if should_skip:
                continue

            # FORMULA PRIORITY CHECK
            # Cambria Math-heavy blocks have already been rendered as images.
            # Skip the corresponding text block so the formula is not emitted twice.
            for formula_bbox_tuple in reference_bboxes.get("formula_blocks", []):
                if (
                    not isinstance(formula_bbox_tuple, (list, tuple))
                    or len(formula_bbox_tuple) != 4
                ):
                    continue

                fx0, fy0, fx1, fy1 = formula_bbox_tuple
                formula_bbox = BoundingBox.from_tuple((fx0, fy0, fx1, fy1))

                contained = (
                    block_bbox.x0 >= formula_bbox.x0
                    and block_bbox.y0 >= formula_bbox.y0
                    and block_bbox.x1 <= formula_bbox.x1
                    and block_bbox.y1 <= formula_bbox.y1
                )
                overlap_pct, _, has_overlap = calculate_bbox_overlap(
                    block_bbox, formula_bbox
                )

                if contained or (has_overlap and overlap_pct > 0.9):
                    should_skip = True
                    break

            if should_skip:
                continue

            text = text.replace("\u00ad", "")  # SOFT HYPHEN
            text = text.replace("\u200b", "")  # Remove zero-width space
            text = text.replace("\u2009", " ")  # Normalize thin space to space
            text = text.replace("\u00a0", " ")  # NBSP
            text = text.replace("\u2002", " ")  # EN SPACE
            text = text.replace("\u200a", " ")  # HAIR SPACE

            if self.config.hyphen_fix_enabled:
                fixed_text, fix_count = fix_hyphenated_lines(text)
            else:
                fixed_text, fix_count = text, 0

            fixed_text = fixed_text.strip()
            if not fixed_text:
                continue

            reference_bboxes["text_blocks"].append(block_bbox_tuple)
            content.append((block_bbox_tuple, fixed_text, fix_count))

        return content

    def _extract_drawing_text_blocks(
        self,
        page: pymupdf.Page,
        drawing_list: List[DrawingInfo],
    ) -> List[Tuple[Tuple[float, float, float, float], str, int]]:
        """Extract each saved drawing's text as one independent text block.

        Text is clipped to the complete drawing bbox, normalized using the same
        rules as normal text, and returned in drawing layout order.  Nothing is
        emitted here; the caller emits these blocks after all plain text.
        """
        content: List[Tuple[Tuple[float, float, float, float], str, int]] = []

        for drawing_info in drawing_list:
            if not drawing_info.saved or drawing_info.bbox is None:
                continue

            drawing_bbox = drawing_info.bbox.to_tuple()
            clip = pymupdf.Rect(drawing_bbox)
            raw_text = page.get_text("text", clip=clip)

            raw_text = raw_text.replace("\u00ad", "")
            raw_text = raw_text.replace("\u200b", "")
            raw_text = raw_text.replace("\u2009", " ")
            raw_text = raw_text.replace("\u00a0", " ")
            raw_text = raw_text.replace("\u2002", " ")
            raw_text = raw_text.replace("\u200a", " ")

            if self.config.hyphen_fix_enabled:
                fixed_text, fix_count = fix_hyphenated_lines(raw_text)
            else:
                fixed_text, fix_count = raw_text, 0

            fixed_text = fixed_text.strip()
            if fixed_text:
                content.append((drawing_bbox, fixed_text, fix_count))

        # Drawings are emitted as whole blocks, ordered by page position.
        content.sort(key=lambda item: (item[0][1], item[0][0]))
        return content

    @staticmethod
    def _text_block_overlaps_drawing(
        block_bbox_tuple: Tuple[float, float, float, float],
        drawing_bboxes: List[Tuple[float, float, float, float]],
    ) -> bool:
        """Return True when any part of a plain text block overlaps a drawing area."""
        block_bbox = BoundingBox.from_tuple(block_bbox_tuple)

        for drawing_bbox_tuple in drawing_bboxes:
            drawing_bbox = BoundingBox.from_tuple(drawing_bbox_tuple)
            _, _, has_overlap = calculate_bbox_overlap(block_bbox, drawing_bbox)
            if has_overlap:
                return True

        return False
    def _record_text_counts(self, text: str) -> None:
        """Record word and character counts for extracted text."""
        self.stats.word_count += count_words(text)
        self.stats.char_count += count_characters(text)

    def _record_table_counts(self, table_data: TableData) -> None:
        """Record counts for useful table content without counting JSON syntax."""
        parts: List[str] = []
        for value in (
            table_data.headline,
            table_data.description,
            table_data.column_headers,
            table_data.data_rows,
        ):
            self._collect_text_values(value, parts)
        self._record_text_counts(" ".join(parts))

    def _collect_text_values(self, value: Any, parts: List[str]) -> None:
        """Flatten nested table values into countable text fragments."""
        if value is None:
            return
        if isinstance(value, str):
            parts.append(value)
            return
        if isinstance(value, dict):
            for key, nested_value in value.items():
                self._collect_text_values(key, parts)
                self._collect_text_values(nested_value, parts)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                self._collect_text_values(item, parts)
            return
        parts.append(str(value))

    # ----------------------------------------------------------------------
    # Layout‑sorting utilities
    # ----------------------------------------------------------------------
    def _sort_blocks_by_layout(
        self,
        blocks: List[Tuple],
        page: Optional[pymupdf.Page] = None,
    ) -> List[Tuple]:
        """Sort text blocks into reading order while respecting column and row structure.

        The algorithm:

        1. Discard blocks smaller than ``min_block_size_px``.
        2. If only one block remains, return it.
        3. Otherwise, detect column boundaries using ``col_tolerance`` (as a
           fraction of page width).
        4. Within each column, group blocks into rows using ``row_tolerance``
           (as a fraction of page height) and sort left‑to‑right inside each row.
        5. Append any non‑column blocks in Y order.
        """
        if not blocks or len(blocks) < 2:
            return [
                block
                for block in blocks
                if passes_size_filter(BoundingBox.from_tuple(block[:4]), self.config.min_block_size_px)
            ]

        # ---- Size filter ----------------------------------------------------
        filtered_blocks = [
            block
            for block in blocks
            if passes_size_filter(BoundingBox.from_tuple(block[:4]), self.config.min_block_size_px)
        ]
        if not filtered_blocks:
            return []

        # ---- Column detection -----------------------------------------------
        blocks_by_x = sorted(filtered_blocks, key=lambda b: b[0])
        columns = self._detect_columns(blocks_by_x, page.rect.width if page is not None else 0.0)

        # ---- Re‑assemble final order -----------------------------------------
        page_height = page.rect.height if page is not None else 0.0
        row_threshold = self.config.row_tolerance * page_height

        sorted_columns: List[List[Tuple]] = []
        all_column_block_indices = set()
        for column in columns:
            # Sort by top edge, then group into rows using row_tolerance.
            column_by_y = sorted(column, key=lambda b: b[1])
            rows: List[List[Tuple]] = []
            current_row: List[Tuple] = [column_by_y[0]]
            for block in column_by_y[1:]:
                if row_threshold > 0 and abs(block[1] - current_row[-1][1]) <= row_threshold:
                    current_row.append(block)
                else:
                    rows.append(sorted(current_row, key=lambda b: b[0]))
                    current_row = [block]
            rows.append(sorted(current_row, key=lambda b: b[0]))

            for row in rows:
                for block in row:
                    all_column_block_indices.add(id(block))
            sorted_columns.extend(rows)

        # Blocks that do not belong to any detected column keep their original Y order
        non_column_blocks = [b for b in filtered_blocks if id(b) not in all_column_block_indices]
        non_column_blocks.sort(key=lambda b: b[1])

        sorted_blocks = []
        sorted_blocks.extend(non_column_blocks)
        for row in sorted_columns:
            sorted_blocks.extend(row)
        return sorted_blocks
    

# In converter.py

    def _detect_columns(
        self,
        blocks_by_x: List[Tuple],
        page_width: float,
    ) -> List[List[Tuple]]:
        """Group blocks that belong to the same vertical column.

        Two blocks are considered part of the same column if:
        1. Their center X coordinates differ by at most ``col_tolerance * page_width``, OR
        2. Their horizontal bounding boxes overlap by more than 50% of their smaller width.

        This is more robust than comparing only left edges, as it handles indented
        text, centered paragraphs, and justified text better.
        """
        if not blocks_by_x:
            return []

        # Pre-compute center X for each block to avoid repeated calculation
        # blocks_by_x items are tuples: (x0, y0, x1, y1, text, ...)
        block_centers = [(block[0] + block[2]) / 2.0 for block in blocks_by_x]
        
        columns: List[List[Tuple]] = []
        used_block_indices: set[int] = set()

        # Sort by center X to make column detection more stable
        sorted_indices = sorted(range(len(blocks_by_x)), key=lambda i: block_centers[i])

        for i_idx, i in enumerate(sorted_indices):
            if i in used_block_indices:
                continue

            current_col = [blocks_by_x[i]]
            used_block_indices.add(i)
            center_i = block_centers[i]
            
            # Check against all other blocks to find those in the same column
            for j_idx, j in enumerate(sorted_indices):
                if j == i or j in used_block_indices:
                    continue
                
                center_j = block_centers[j]
                
                # Criterion 1: Center X alignment within tolerance
                col_threshold = self.config.col_tolerance * page_width
                centers_close = abs(center_i - center_j) <= col_threshold
                
                # Criterion 2: Significant horizontal overlap (fallback for indented/justified text)
                # Calculate horizontal overlap percentage relative to the smaller width
                i_x0, _, i_x1, _ = blocks_by_x[i][:4]
                j_x0, _, j_x1, _ = blocks_by_x[j][:4]
                
                min_width = max(1e-6, min(i_x1 - i_x0, j_x1 - j_x0)) # Avoid division by zero
                overlap_start = max(i_x0, j_x0)
                overlap_end = min(i_x1, j_x1)
                overlap_width = max(0.0, overlap_end - overlap_start)
                
                overlap_ratio = overlap_width / min_width
                overlaps_significantly = overlap_ratio > 0.7

                if centers_close or overlaps_significantly:
                    current_col.append(blocks_by_x[j])
                    used_block_indices.add(j)

            columns.append(current_col)

        return columns


    # ----------------------------------------------------------------------
    # Media metadata generation
    # ----------------------------------------------------------------------
    def _write_formula_info(
        self,
        formula_list: List[FormulaInfo],
        page_num: int,
    ) -> List[str]:
        """Create a JSON block describing Cambria Math formula images."""
        content: List[str] = []

        if formula_list:
            saved_count = sum(1 for formula in formula_list if formula.saved)
            self.stats.formulas_saved += saved_count

            formula_data = {
                "formula_count": len(formula_list),
                "formulas_saved": saved_count,
                "font_name": self.config.formula_font_name,
                #"font_threshold": self.config.formula_font_threshold,
                "media_folder": f"{self.pdf_filename}_media",
                "formulas_on_page": [
                    {
                        "number": formula.formula_num,
                        "page_number": formula.page_num,
                        #"bbox": formula.bbox, # optional
                        #"char_count": formula.char_count, # optional
                        #"cambria_math_char_count": formula.cambria_math_char_count, # optional
                        #"cambria_math_ratio": formula.cambria_math_ratio, # optional
                        "saved": formula.saved,
                        "filename": formula.filename if formula.saved else None,
                        "skipped_reason": formula.skipped_reason,
                    }
                    for formula in formula_list
                ],
            }

            content.append(
                f"\n\nFORMULAS FOUND ON PAGE {page_num}: {len(formula_list)}\n\n"
            )
            content.append("```json\n")
            content.append(json.dumps(formula_data, indent=2, ensure_ascii=False))
            content.append("\n```\n\n")

        return content

    def _write_image_info(self, images_on_page: List[ImageInfo], page_num: int) -> List[str]:
        """Create a JSON block that lists all images found on the page.

        If images are saved, counters are updated in ``self.stats``.  The JSON
        output contains resolution, saved flag, filename, and any skip reason.
        """
        content = []
        if images_on_page:
            saved_count = sum(1 for img in images_on_page if img.saved)
            skipped_count = len(images_on_page) - saved_count
            self.stats.images_saved += saved_count
            self.stats.images_skipped += skipped_count

            image_data = {
                "image_count": len(images_on_page),
                "images_saved": saved_count,
                "images_skipped_small": skipped_count,
                "min_size_threshold": self.config.min_size_px,
                "media_folder": f"{self.pdf_filename}_media",
                "images_with_resolution": [
                    {
                        "index": img.img_idx,
                        "width": img.width,
                        "height": img.height,
                        "saved": img.saved,
                        "filename": img.filename,
                        "skipped_reason": img.skipped_reason,
                    }
                    for img in images_on_page
                    if img.saved
                ],
            }

            content.append(f"\n\nIMAGES FOUND ON PAGE {page_num}: {len(images_on_page)}\n\n")
            content.append("```json\n")
            content.append(json.dumps(image_data, indent=2, ensure_ascii=False))
            content.append("\n```\n\n")

        return content

    def _write_drawing_info(self, drawing_list: List[DrawingInfo], page_num: int) -> List[str]:
        """Create drawing JSON only when drawing-image saving is enabled."""
        content = []

        # Drawings are always detected internally, but drawing JSON is an output
        # feature controlled strictly by the save_drawings flag. When disabled,
        # no drawing JSON is emitted even if drawings were detected.
        if not self.config.save_drawings:
            return content

        # A drawing may be detected but later fail to save (or be removed by
        # image/drawing conflict resolution). Only successfully saved images
        # should produce drawing JSON metadata.
        saved_drawings = [
            drawing
            for drawing in drawing_list
            if drawing.saved and drawing.filename
        ]

        if not saved_drawings:
            return content

        drawings_saved = len(saved_drawings)
        self.stats.drawings_saved += drawings_saved

        drawing_data = {
            "drawing_count": drawings_saved,
            "drawings_saved": drawings_saved,
            "minimum_size_threshold_px": self.config.min_size_px,
            "media_folder": f"{self.pdf_filename}_media",
            "drawings_with_resolution": [
                {
                    "index": drawing.index,
                    "resolution": drawing.resolution,
                    "saved": True,
                    "filename": drawing.filename,
                    "skipped_reason": None,
                }
                for drawing in saved_drawings
            ],
        }

        content.append(f"\n\nDRAWINGS SAVED ON PAGE {page_num}: {drawings_saved}\n\n")
        content.append("```json\n")
        content.append(json.dumps(drawing_data, indent=2, ensure_ascii=False))
        content.append("\n```\n\n")

        return content


# ----------------------------------------------------------------------
# High‑level PDF converter (orchestrates multiprocessing)
# ----------------------------------------------------------------------
class PDFConverter:
    """Main entry point that converts a list of PDFs to plain‑text files.

    The class encapsulates configuration, validation, task creation, and
    multiprocessing execution.  It is deliberately thin – most of the heavy
    lifting lives in ``PageProcessor`` and its helpers.
    """

    def __init__(
        self,
        config: Optional[ConversionConfig] = None,
        progress_callback: Optional[Callable] = None,
        stop_flag: Optional[Any] = None,
    ):
        self.config = config or ConversionConfig()
        self.progress_callback = progress_callback  # (pdf_name, page_done, total, imgs, dws)
        self.stop_flag = stop_flag  # threading.Event – checked between chunks

    def convert_parallel(
        self,
        pdf_files: List[str],
        output_dir: str,
        cores: int = None,
        output_in_source_dir: bool = True,
    ) -> Dict[str, Any]:
        """Convert many PDFs in parallel and return aggregated statistics.

        Steps performed:

        1. Validate each PDF (skipping corrupted or protected files).
        2. Clear any existing ``_media`` folders so stale images/drawings from
           previous runs do not persist.
        3. Build processing tasks based on core count and page count.
        4. Dispatch tasks to a ``ProcessPoolExecutor``.
        5. Assemble per‑page results into a single ``.txt`` file per PDF.
        6. Merge per‑file stats into a global summary.

        When ``output_in_source_dir`` is ``True``, each PDF's .txt and _media
        folder are written next to the original PDF instead of *output_dir*.

        All side‑effects (file writes, multiprocessing) are confined to this
        method.
        """
        if cores is None:
            cores = get_physical_cores()

        valid_files = self._collect_valid_files(pdf_files)

        if not valid_files:
            print("No valid PDFs found")
            return {}

        # Clear stale media folder contents before processing
        self._clear_media_folders(valid_files, output_in_source_dir, output_dir)

        tasks = create_processing_tasks(valid_files, cores)
        
        global_stats = self._create_global_stats()
        results_by_file = self._run_tasks(tasks, output_dir, cores, global_stats)

        # Build a lookup of original source directory per filename
        source_dirs: Dict[str, str] = {}
        for fi in valid_files:
            source_dirs[fi["filename"]] = os.path.dirname(os.path.abspath(fi["path"]))

        metadata_by_file = {
            file_info["filename"]: file_info.get("metadata", {})
            for file_info in valid_files
        }

        # Write each PDF's results and merge per‑file stats back into ``global_stats``
        per_file_stats: Dict[str, Dict[str, int]] = {}
        for pdf_filename, results in results_by_file.items():
            if output_in_source_dir:
                target_dir = source_dirs.get(pdf_filename, output_dir)
            else:
                target_dir = output_dir
            output_file = os.path.join(target_dir, f"{pdf_filename}.txt")
            media_dir = os.path.join(target_dir, f"{pdf_filename}_media")
            file_stats = assemble_results(
                results,
                pdf_filename,
                output_file,
                media_dir,
                metadata=metadata_by_file.get(pdf_filename),
                config=self.config,
            )
            per_file_stats[pdf_filename] = file_stats
            self._merge_file_stats(global_stats, file_stats)

        global_stats["per_file"] = per_file_stats
        return global_stats

    def _collect_valid_files(self, pdf_files: List[str]) -> List[Dict[str, Any]]:
        """Return only those PDFs that can be opened and are not encrypted.

        Each entry contains the absolute path, page count, and a stem‑based
        filename used for output naming.
        """
        valid_files = []
        for pdf_path in pdf_files:
            result = PDFValidator.validate_file(pdf_path)
            if not result["valid"]:
                print(f"Skipping invalid file {Path(pdf_path).name}: {result['reason']}")
                continue

            valid_files.append(
                {
                    "path": pdf_path,
                    "page_count": result["page_count"],
                    "filename": Path(pdf_path).stem,
                    "metadata": result.get("metadata", {}),
                }
            )
        return valid_files


    def _create_global_stats(self) -> Dict[str, int]:
        """Initialize a dict that will collect aggregates across all files."""
        return {
            "files_processed": 0,
            "total_pages": 0,
            "images_saved": 0,
            "drawings_saved": 0,
            "hyphens_fixed": 0,
            "table_hyphens_fixed": 0,
            "word_count": 0,
            "char_count": 0,
            "formulas_saved": 0,
            "errors": 0,
        }

    def _run_tasks(
        self,
        tasks: List[Dict[str, Any]],
        output_dir: str,
        cores: int,
        global_stats: Dict[str, int],
    ) -> Dict[str, List[Tuple[int, str, Dict]]]:
        """Execute all page‑level tasks in a process pool.

        The function builds a ``future_to_task`` mapping, submits each page
        job, and then collects the results.  Results are grouped by the PDF
        filename they belong to.
        """
        config_dict = {k: v for k, v in self.config.__dict__.items()}
        future_to_task: Dict[Any, Dict[str, Any]] = {}
        results_by_file: Dict[str, List[Tuple[int, str, Dict]]] = {}

        # ---- compute total page budget (for progress reporting) ----
        total_pages: int = 0
        for task in tasks:
            total_pages += (task["page_range"][1] - task["page_range"][0] + 1)

        with ProcessPoolExecutor(max_workers=cores) as executor:
            for task in tasks:
                # stop submitting new chunks as soon as cancellation is requested
                if self.stop_flag is not None and self.stop_flag.is_set():
                    break
                pdf_path = task["pdf_path"]
                pdf_filename = task["pdf_filename"]
                page_range = task["page_range"]
                total_pages_this = task.get("total_page_count", page_range[1])
                media_dir = os.path.join(os.path.dirname(os.path.abspath(pdf_path)), f"{pdf_filename}_media")

                future = executor.submit(
                    _process_pages_worker,
                    pdf_path, page_range, total_pages_this,
                    config_dict, media_dir, pdf_filename,
                )
                future_to_task[future] = {
                    "pdf_filename": pdf_filename,
                    "page_range": page_range,
                    "expected_pages": (page_range[1] - page_range[0] + 1),
                }

            # ---- Collect results -------------------------------------------------
            pages_done = 0
            pages_done_by_file: Dict[str, int] = {}
            imgs_by_file: Dict[str, int] = {}
            dws_by_file: Dict[str, int] = {}
            formulas_by_file: Dict[str, int] = {}
            cancelled = False
            for future in as_completed(future_to_task):
                # honour a cancellation request: cancel every not-yet-started
                # future once, then keep draining the already-running ones
                if not cancelled and self.stop_flag is not None and self.stop_flag.is_set():
                    cancelled = True
                    for pending in future_to_task:
                        pending.cancel()
                if future.cancelled():
                    continue
                task_info = future_to_task[future]
                try:
                    chunk_results = future.result()
                    filename = task_info["pdf_filename"]
                    if filename not in results_by_file:
                        results_by_file[filename] = []
                    for page_num, content, stats in chunk_results:
                        results_by_file[filename].append((page_num, content, stats))
                        result_pages = stats.get("pages_attempted", 1)
                        try:
                            result_pages = int(result_pages)
                        except (TypeError, ValueError):
                            result_pages = 1
                        result_pages = max(result_pages, 1)
                        pages_done += result_pages
                        pages_done_by_file[filename] = (
                            pages_done_by_file.get(filename, 0) + result_pages
                        )
                        imgs_by_file[filename] = imgs_by_file.get(filename, 0) + stats.get("images_saved", 0)
                        dws_by_file[filename] = dws_by_file.get(filename, 0) + stats.get("drawings_saved", 0)
                        formulas_by_file[filename] = formulas_by_file.get(filename, 0) + stats.get("formulas_saved", 0)
                        # fire progress callback with per-file cumulative counts
                        if self.progress_callback:
                            self.progress_callback(
                                pdf_filename=filename,
                                pages_done=pages_done_by_file[filename],
                                total_pages=total_pages,
                                images_saved=imgs_by_file[filename],
                                drawings_saved=dws_by_file[filename],
                                formulas_saved=formulas_by_file[filename],
                            )
                except Exception as exc:
                    print(f"Error processing chunk {task_info['page_range']}: {exc}")
                    global_stats["errors"] += task_info.get("expected_pages", 1)

        return results_by_file

    def _merge_file_stats(self, global_stats: Dict[str, int], file_stats: Dict[str, int]) -> None:
        """Update the aggregated stats with per‑file counters."""
        global_stats["files_processed"] += 1
        global_stats["total_pages"] += file_stats["pages_processed"]
        global_stats["images_saved"] += file_stats["images_saved"]
        global_stats["drawings_saved"] += file_stats["drawings_saved"]
        global_stats["hyphens_fixed"] += file_stats["hyphens_fixed"]
        global_stats["table_hyphens_fixed"] += file_stats["table_hyphens_fixed"]
        global_stats["word_count"] += file_stats["word_count"]
        global_stats["char_count"] += file_stats["char_count"]
        global_stats["formulas_saved"] += file_stats["formulas_saved"]
        global_stats["errors"] += file_stats["errors"]

    def _clear_media_folders(
        self,
        valid_files: List[Dict[str, Any]],
        output_in_source_dir: bool,
        output_dir: str,
    ) -> None:
        """Previously this cleared stale media-folder contents before a new
        conversion. That behaviour was removed: old images and drawings must
        never be deleted. New assets are written into the existing media
        folder and naturally overwrite files with the same name; files that
        are not regenerated are left untouched.

        Kept as an empty stub so the call site in ``convert_parallel`` does
        not need to change.
        """
        pass


# ----------------------------------------------------------------------
# Configuration factory and CLI entry point
# ----------------------------------------------------------------------
def build_default_config(
    no_images: bool = False,
    no_drawings: bool = False,
    no_hyphen_fix: bool = False,
    include_metadata: bool = True,
    no_formulas: bool = False,
) -> ConversionConfig:
    """Create a sensible default ``ConversionConfig`` for command‑line use.

    Parameters
    ----------
    no_images, no_drawings, no_hyphen_fix, no_formulas : bool
        When ``True`` the corresponding extraction/cleanup step is disabled.
    include_metadata : bool
        When ``True`` PDF document metadata is written once at the top of
        each output file.
    """
    return ConversionConfig(
        save_images=not no_images,
        save_drawings=not no_drawings,
        min_size_px=100,
        bbox_padding=20.0,
        row_tolerance=0.0195,
        col_tolerance=0.016,
        cluster_merge_tolerance=25.0,
        max_items_per_page=10,
        min_items_per_cluster=10,
        include_small_text_blocks=True,
        small_text_max_chars=50,
        enable_margin_check=True,
        hyphen_fix_enabled=not no_hyphen_fix,
        include_metadata=include_metadata,
    )


def run_full_conversion(
    pdf_files: List[str],
    output_folder: str,
    cores: int,
    config: ConversionConfig,
) -> Dict[str, Any]:
    """Run the conversion pipeline and print a timing summary.

    This function is typically used by a CLI wrapper; it measures elapsed time,
    pages processed, and pages‑per‑second, then returns the global stats dict.
    """
    start_time = time.time()
    converter = PDFConverter(config)
    global_stats = converter.convert_parallel(pdf_files, output_folder, cores)
    elapsed_time = time.time() - start_time

    pages = global_stats.get("total_pages", 0)
    pages_per_second = pages / elapsed_time if elapsed_time > 0 else 0
    print(f"Total Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"Pages Processed: {pages}")
    print(f"Pages per Second: {pages_per_second:.2f}")
    return global_stats



