"""Table extraction cleanup and structuring utilities."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from overlap import (
    combine_nonempty_texts,
    horizontal_overlap_width,
    is_close_above_table,
    is_close_below_table,
    is_small_text_block,
)
from layout import BoundingBox



from post_process import fix_hyphenated_lines, remove_nonprintable_chars

# Placeholder cell values treated as empty (case-insensitive).
_PLACEHOLDER_VALUES = ("", "null", "none", "nan", "-", "_")


@dataclass
class TableData:
    """Processed table data with metadata."""
    # Type of table: 'headline', 'standard', or 'no_data'
    type: str
    # Optional headline text (e.g., single-cell top row)
    headline: Optional[str]
    # Optional description text (e.g., single-cell bottom row)
    description: Optional[str]
    # List of column header names (cleaned and deduplicated)
    column_headers: List[str]
    # List of data rows, each as a dict mapping column headers to cell values
    data_rows: List[Dict[str, Any]]
    # Number of actual data rows (excluding header and empty rows)
    data_rows_count: int
    # Type of header structure: 'standard_headers', 'corner_empty_table', or 'vertical_headers'
    header_structure_type: str
    # Whether the table has an empty top-left cell (i.e., corner is empty)
    is_corner_empty_table: bool
    # Whether the first row is recognized as a header row
    has_header_row: bool


class TableProcessor:
    """Process and validate table data from PDF."""

    @staticmethod
    def make_raw_table(
        table: Any,
        raw_data: List[List[Any]],
        page_num: Optional[int] = None,
        table_num: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Convert a PyMuPDF Table into a JSON-serializable raw representation."""
        cells = [
            list(cell) if cell is not None else None
            for cell in table.cells
        ]
        return {
            "table_number": table_num,
            "page_number": page_num,
            "bbox": list(table.bbox),
            "row_count": int(table.row_count),
            "column_count": int(table.col_count),
            "cells": cells,
            "extracted_data": raw_data,
        }

    @staticmethod
    def prepare_table_data(
        raw_table: Dict[str, Any],
        hyphen_fix_enabled: bool = True,
        return_fix_count: bool = False,
    ):
        """
        Reconstruct merged cells from bbox geometry, then clean values.

        Headline/description candidates are detected from the original extracted
        data before merged-cell expansion. Only the first and last rows can be
        special rows, and only when exactly one cell contains meaningful content.
        """
        raw_data = raw_table.get("extracted_data") or []

        matrix = [
            list(row) if isinstance(row, (list, tuple)) else [row]
            for row in raw_data
        ]

        headline_text = None
        description_text = None
        headline_row_idx = None
        description_row_idx = None

        # Only the FIRST row can be a headline.
        if matrix:
            headline_text = TableProcessor._get_single_content_cell(matrix[0])
            if headline_text is not None:
                headline_row_idx = 0

        # Only the LAST row can be a description.
        # Do not classify the same row as both.
        if len(matrix) > 1:
            description_text = TableProcessor._get_single_content_cell(matrix[-1])
            if description_text is not None:
                description_row_idx = len(matrix) - 1

        # Expand the ORIGINAL table unchanged so bbox geometry remains valid.
        expanded = TableProcessor.expand_merged_cells(raw_table)

        # Restore special rows as single-content rows after expansion.
        # This prevents merged-cell propagation from turning a headline or
        # description into a normal header/data row.
        if headline_row_idx is not None and headline_row_idx < len(expanded):
            expanded[headline_row_idx] = [None] * len(expanded[headline_row_idx])
            expanded[headline_row_idx][0] = headline_text

        if description_row_idx is not None and description_row_idx < len(expanded):
            expanded[description_row_idx] = [None] * len(expanded[description_row_idx])
            expanded[description_row_idx][0] = description_text

        return TableProcessor.clean_table_data(
            expanded,
            hyphen_fix_enabled=hyphen_fix_enabled,
            return_fix_count=return_fix_count,
        )

    @staticmethod
    def clean_table_data(
        table_data: Union[List, Tuple],
        hyphen_fix_enabled: bool = True,
        return_fix_count: bool = False,
    ) -> Union[List[Any], Tuple[List[Any], int]]:
        """
        Clean all table cell values:
        - Remove non-printable characters
        - Optionally fix hyphenated line breaks (e.g., 'ex-\nample' -> 'example')
        - Preserve non-string values (e.g., numbers, None)
        """
        cleaned = []
        fix_count = 0

        def clean_cell(cell: Any) -> Any:
            nonlocal fix_count
            if not isinstance(cell, str):
                return cell

            if hyphen_fix_enabled:
                cell, cell_fix_count = fix_hyphenated_lines(cell)
                fix_count += cell_fix_count

            return remove_nonprintable_chars(cell)

        for row in table_data:
            if isinstance(row, (list, tuple)):
                cleaned_row = [clean_cell(cell) for cell in row]
                cleaned.append(cleaned_row)
            elif isinstance(row, str):
                cleaned.append(clean_cell(row))
            else:
                cleaned.append(row)

        if return_fix_count:
            return cleaned, fix_count
        return cleaned

    @staticmethod
    def is_useful_table(table_data: List[List]) -> bool:
        """
        Validate if detected table contains useful structure.
        Filter out:
        - Empty tables
        - Tables with >75% empty rows
        - Tables with <3 meaningful cells
        - Tables with only 1 row (headers-only or single-row lists)
        - Tables with only 1 column (single-column lists)
        """
        if not table_data or len(table_data) == 0:
            return False

        list_rows = [row for row in table_data if isinstance(row, list)]
        if not list_rows:
            return False

        # Reject tables with only 1 row (e.g., header-only, no data rows)
        if len(list_rows) <= 1:
            return False

        # Count total cells across all list rows
        total_cells = sum(len(row) for row in list_rows)
        if total_cells == 0:
            return False

        # Count rows that are entirely empty/placeholder
        empty_row_count = 0
        for row in list_rows:
            is_empty_row = True
            for cell in row:
                cell_str = str(cell).strip().lower() if cell else ""
                if cell_str not in ("", "null", "none", "nan", "-", "_"):
                    is_empty_row = False
                    break
            if is_empty_row:
                empty_row_count += 1

        # Reject if >75% rows are empty
        if empty_row_count / len(list_rows) > 0.75:
            return False

        # Count non-empty, non-placeholder cells
        non_empty_cells = total_cells - sum(
            len([cell for cell in row if str(cell).strip().lower() in ("", "null", "none")])
            for row in list_rows
        )
        if non_empty_cells < 3:
            return False

        # Determine the max number of columns across all rows
        max_cols = max((len(row) for row in list_rows), default=0)

        # Reject tables with only 1 column (single-column lists are not useful tables)
        if max_cols <= 1:
            return False

        return True

    @staticmethod
    def process_table(
        table_data: List[List[str]],
        page_num: int = None,
        table_num: int = None,
    ) -> TableData:
        """
        Process raw table data into structured TableData.

        Steps:
        1. Detect and extract optional headline/description (single-content rows at top/bottom).
        2. Determine if first row is header, and whether it's a corner-empty table.
        3. Clean and deduplicate headers.
        4. Expand merged cells from raw bbox geometry.
        5. Process data rows accordingly (with or without corner-empty logic).
        """

        # Handle empty input
        if not table_data or len(table_data) < 1:
            return TableData(
                type="no_data",
                headline=None,
                description=None,
                column_headers=[],
                data_rows=[],
                data_rows_count=0,
                header_structure_type="standard_headers",
                is_corner_empty_table=False,
                has_header_row=False,
            )

        # Initialize metadata
        headline_text = None
        description_text = None
        rows_for_processing = list(table_data)  # Copy to avoid mutation

        # Detect and extract headline.
        # The meaningful cell can be in any column, not necessarily column 0.
        if rows_for_processing:
            headline_text = TableProcessor._get_single_content_cell(
                rows_for_processing[0]
            )

            if headline_text is not None:
                del rows_for_processing[0]

        # Detect and extract description.
        # The meaningful cell can be in any column, not necessarily column 0.
        if rows_for_processing:
            description_text = TableProcessor._get_single_content_cell(
                rows_for_processing[-1]
            )

            if description_text is not None:
                del rows_for_processing[-1]

        has_header_row = False
        header_structure_type = "standard_headers"
        headers: List[str] = []
        is_corner_empty = False
        data_rows: List[Dict[str, Any]] = []

        # Two-row tables (with non-empty corner) use the first column as headers.
        if (
            len(rows_for_processing) == 2
            and all(isinstance(row, list) for row in rows_for_processing)
            and len(rows_for_processing[0]) >= 3
            and len(rows_for_processing[1]) >= 3
            and not TableProcessor._is_empty_cell(rows_for_processing[0][0])
        ):
            header_structure_type = "vertical_headers"
            has_header_row = True

            row_1, row_2 = rows_for_processing
            headers = [
                str(row_1[0]).strip(),
                str(row_2[0]).strip(),
            ]

            col_count = max(len(row_1), len(row_2))
            for col_idx in range(1, col_count):
                cell_1 = TableProcessor._cell_value(row_1, col_idx)
                cell_2 = TableProcessor._cell_value(row_2, col_idx)
                if cell_1 and cell_2:
                    data_rows.append(
                        {
                            headers[0]: cell_1,
                            headers[1]: cell_2,
                        }
                    )

        # Try to detect header row (first remaining row)
        elif len(rows_for_processing) > 0 and isinstance(rows_for_processing[0], list):
            potential_header_row = rows_for_processing[0]
            first_cell_empty = (
                len(potential_header_row) >= 2
                and str(potential_header_row[0]).strip() in ("", "null", "none", "-", "_")
            )

            if first_cell_empty:
                is_corner_empty = True
                header_structure_type = "corner_empty_table"
                has_header_row = True
                raw_headers = [
                    str(cell).strip()
                    for cell in potential_header_row[1:]
                    if str(cell).strip() and str(cell).strip().lower() not in ("null", "none")
                ]
            else:
                has_header_row = True
                raw_headers = [
                    str(cell).strip()
                    for cell in potential_header_row
                    if str(cell).strip() and str(cell).strip().lower() not in ("null", "none")
                ]

            # Deduplicate headers by appending suffixes (e.g., "Name", "Name_1")
            seen_headers: Dict[str, int] = {}
            for header in raw_headers:
                if header in seen_headers:
                    seen_headers[header] += 1
                    headers.append(f"{header}_{seen_headers[header]}")
                else:
                    seen_headers[header] = 0
                    headers.append(header)

            # Process data rows.
            data_start_idx = 1 if has_header_row else 0

            for current_row in rows_for_processing[data_start_idx:]:
                if not isinstance(current_row, list):
                    continue

                if is_corner_empty and len(headers) > 0:
                    row_label = str(current_row[0]).strip()

                    if not row_label or row_label.lower() in ("null", "none"):
                        continue

                    filtered_dict = {}
                    for j, col_header in enumerate(headers):
                        data_col_idx = 1 + j

                        if data_col_idx < len(current_row):
                            cell_val = str(current_row[data_col_idx]).strip()
                            if cell_val and cell_val.lower() not in ("null", "none"):
                                filtered_dict[col_header] = cell_val

                    if filtered_dict:
                        data_rows.append({row_label: filtered_dict})

                else:
                    filtered_row = {}

                    for j, col_header in enumerate(headers):
                        if j < len(current_row):
                            cell_val = str(current_row[j]).strip()
                            if cell_val and cell_val.lower() not in ("null", "none"):
                                filtered_row[col_header] = cell_val

                    if filtered_row:
                        data_rows.append(filtered_row)

            data_rows_count = len(data_rows)

        return TableData(
            type="headline" if headline_text else "standard",
            headline=headline_text,
            description=description_text,
            column_headers=headers,
            data_rows=data_rows,
            data_rows_count=len(data_rows),
            header_structure_type=header_structure_type,
            is_corner_empty_table=is_corner_empty,
            has_header_row=has_header_row,
        )

    @staticmethod
    def _cluster_coordinates(
        values: List[float],
        tolerance: float = 0.05,
    ) -> List[float]:
        """Merge nearly identical PDF coordinates into logical boundaries."""
        result: List[float] = []

        for value in sorted(values):
            if not result or abs(value - result[-1]) > tolerance:
                result.append(value)
            else:
                result[-1] = (result[-1] + value) / 2.0

        return result

    @staticmethod
    def _boundary_index(
        value: float,
        boundaries: List[float],
        tolerance: float = 0.05,
    ) -> int:
        """Map a PDF coordinate to the nearest logical boundary."""
        index = min(
            range(len(boundaries)),
            key=lambda i: abs(boundaries[i] - value),
        )

        if abs(boundaries[index] - value) > tolerance:
            raise ValueError(
                f"Cannot map PDF coordinate {value} to logical boundary"
            )

        return index

# In tables.py, modify the expand_merged_cells method:

    @staticmethod
    def expand_merged_cells(
        raw_table: Dict[str, Any],
        values: Optional[List[List[Any]]] = None,
    ) -> List[List[Any]]:
        """
        Reconstruct the logical table from parser bboxes.

        A merged PDF cell appears as one bbox covering multiple logical
        row/column positions. Its value is stored at the top-left logical
        position in extracted_data and is propagated over the full bbox.
        
        If overlapping cells are detected, the function falls back to the 
        original extracted data matrix to avoid crashing.
        """
        cells = raw_table.get("cells") or []
        data = values if values is not None else (raw_table.get("extracted_data") or [])

        rows = int(raw_table.get("row_count", len(data)))
        cols = int(raw_table.get("column_count", 0))

        if rows == 0 or cols == 0:
            return []

        # Normalize extracted_data to a rectangular matrix.
        matrix = [
            list(row) if isinstance(row, (list, tuple)) else [row]
            for row in data
        ]
        while len(matrix) < rows:
            matrix.append([])

        matrix = [
            row[:cols] + [None] * max(0, cols - len(row))
            for row in matrix[:rows]
        ]

        if not cells:
            return matrix

        table_bbox = raw_table.get("bbox")
        if not table_bbox or len(table_bbox) != 4:
            return matrix  # Fallback to original matrix if bbox is missing

        # Build the logical grid from every bbox boundary.
        xs = [table_bbox[0], table_bbox[2]]
        ys = [table_bbox[1], table_bbox[3]]

        for x1, y1, x2, y2 in cells:
            xs.extend((x1, x2))
            ys.extend((y1, y2))

        # Use a slightly larger tolerance to handle floating-point variations
        xs = TableProcessor._cluster_coordinates(xs, tolerance=0.1)
        ys = TableProcessor._cluster_coordinates(ys, tolerance=0.1)

        if len(xs) != cols + 1 or len(ys) != rows + 1:
            # If the grid dimensions don't match, we can't reliably expand.
            # Return the original matrix.
            return matrix

        result = [[None] * cols for _ in range(rows)]
        occupied = [[False] * cols for _ in range(rows)]
        overlap_detected = False

        for bbox in cells:
            x1, y1, x2, y2 = bbox

            try:
                c0 = TableProcessor._boundary_index(x1, xs, tolerance=0.1)
                c1 = TableProcessor._boundary_index(x2, xs, tolerance=0.1) - 1
                r0 = TableProcessor._boundary_index(y1, ys, tolerance=0.1)
                r1 = TableProcessor._boundary_index(y2, ys, tolerance=0.1) - 1
            except (ValueError, IndexError):
                # If mapping fails for any cell, we can't trust the expansion.
                return matrix

            if not (0 <= r0 <= r1 < rows and 0 <= c0 <= c1 < cols):
                # Invalid bounds, skip expansion.
                return matrix

            value = matrix[r0][c0]

            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if occupied[r][c]:
                        # Overlapping cells detected.
                        # Instead of raising an exception, we flag it and break.
                        overlap_detected = True
                        break
                    result[r][c] = value
                    occupied[r][c] = True
                if overlap_detected:
                    break
            
            if overlap_detected:
                break

        if overlap_detected:
            # If we detected overlaps, the expansion is unreliable.
            # Return the original matrix to be safe.
            return matrix

        for r in range(rows):
            for c in range(cols):
                if not occupied[r][c]:
                    result[r][c] = matrix[r][c]

        return result


    @staticmethod
    def _is_empty_cell(cell: Any) -> bool:
        """Return True if the cell is empty or a placeholder value."""
        return str(cell).strip().lower() in _PLACEHOLDER_VALUES

    @staticmethod
    def _cell_value(row: List, index: int) -> str:
        """Return the stripped cell string, or '' if the index is out of range."""
        if index < len(row):
            return str(row[index]).strip()
        return ""

    @staticmethod
    def _get_single_content_cell(row: List[Any]) -> Optional[str]:
        """
        Return the only meaningful cell in a row.

        A row qualifies only when exactly one cell contains meaningful content
        and every other cell is empty or a recognized placeholder.
        """
        if not isinstance(row, list):
            return None

        content_cells = [
            str(cell).strip()
            for cell in row
            if not TableProcessor._is_empty_cell(cell)
        ]

        if len(content_cells) != 1:
            return None

        return content_cells[0]


    @staticmethod
    def _find_special_row(
        raw_table: Dict[str, Any],
        matrix: List[List[Any]],
        from_start: bool,
    ) -> Optional[int]:
        """
        Find a headline/description row from the original table structure.

        A special row is a row where:
        - it contains exactly one meaningful logical cell, OR
        - its content comes from one or more cells spanning the complete
          table width.

        The original bbox geometry is used before merged values are expanded.
        """
        cells = raw_table.get("cells") or []
        bbox = raw_table.get("bbox")

        if not matrix:
            return None

        candidate_indices = (
            range(len(matrix))
            if from_start
            else range(len(matrix) - 1, -1, -1)
        )

        for row_idx in candidate_indices:
            row = matrix[row_idx]

            # First handle the simple case:
            # exactly one meaningful cell in the original extracted row.
            meaningful = [
                cell
                for cell in row
                if not TableProcessor._is_empty_cell(cell)
            ]

            if len(meaningful) == 1:
                return row_idx

            # If bbox information is unavailable, we cannot reliably determine
            # whether repeated values originate from merged cells.
            if not cells or not bbox:
                continue

            if TableProcessor._is_full_width_merged_row(
                raw_table,
                row_idx,
            ):
                return row_idx

        return None


    @staticmethod
    def _is_full_width_merged_row(
        raw_table: Dict[str, Any],
        row_idx: int,
        tolerance: float = 0.05,
    ) -> bool:
        """
        Return True when the physical row is represented by cell geometry
        spanning the complete table width.

        This is checked against the original PDF bboxes, before merged values
        are propagated.
        """
        cells = raw_table.get("cells") or []
        table_bbox = raw_table.get("bbox")
        rows = int(raw_table.get("row_count", 0))
        cols = int(raw_table.get("column_count", 0))

        if (
            not table_bbox
            or len(table_bbox) != 4
            or not cells
            or rows <= 0
            or cols <= 0
        ):
            return False

        xs = [table_bbox[0], table_bbox[2]]
        ys = [table_bbox[1], table_bbox[3]]

        for cell in cells:
            if not cell or len(cell) != 4:
                continue
            x1, y1, x2, y2 = cell
            xs.extend((x1, x2))
            ys.extend((y1, y2))

        xs = TableProcessor._cluster_coordinates(xs, tolerance)
        ys = TableProcessor._cluster_coordinates(ys, tolerance)

        if len(xs) != cols + 1 or len(ys) != rows + 1:
            return False

        try:
            row_top = ys[row_idx]
            row_bottom = ys[row_idx + 1]
        except IndexError:
            return False

        row_cells = []

        for cell in cells:
            if not cell or len(cell) != 4:
                continue

            x1, y1, x2, y2 = cell

            # Cell belongs to this logical row.
            if (
                abs(y1 - row_top) <= tolerance
                and abs(y2 - row_bottom) <= tolerance
            ):
                row_cells.append((x1, x2))

        if not row_cells:
            return False

        # Merge horizontal coverage of cells in this row.
        row_cells.sort()

        current_start, current_end = row_cells[0]

        merged_start = current_start
        merged_end = current_end

        for start, end in row_cells[1:]:
            if start <= merged_end + tolerance:
                merged_end = max(merged_end, end)
            else:
                break

        # The row must span the entire table width.
        return (
            abs(merged_start - table_bbox[0]) <= tolerance
            and abs(merged_end - table_bbox[2]) <= tolerance
        )


    @staticmethod
    def _extract_row_text(row: List[Any]) -> str:
        """
        Extract meaningful text from a special row.

        Duplicate values produced by merged-cell expansion are removed while
        preserving the original order.
        """
        values = []
        seen = set()

        for cell in row:
            if TableProcessor._is_empty_cell(cell):
                continue

            value = str(cell).strip()
            if not value:
                continue

            if value not in seen:
                seen.add(value)
                values.append(value)

        return " ".join(values)


    @staticmethod
    def attach_nearby_text_blocks(
        table_data: "TableData",
        table_bbox: BoundingBox,
        text_blocks: List[Tuple],
        used_attached_bboxes: set,
        max_lines: int = 2,
        max_chars: int = 250,
        max_distance_px: float = 10.0,
        max_overlap_px: float = 2.0,
    ) -> "TableData":
        """
        Attach small nearby text blocks to a table.

        Text blocks immediately above the table are merged into
        ``table_data.headline``. Text blocks immediately below the table are
        merged into ``table_data.description``.

        Consumed text block bboxes are added to ``used_attached_bboxes`` so the
        caller can omit them from normal text extraction.
        """
        best_above_bbox: Optional[Tuple[float, float, float, float]] = None
        best_above_text: Optional[str] = None
        best_above_gap: float = float("inf")

        best_below_bbox: Optional[Tuple[float, float, float, float]] = None
        best_below_text: Optional[str] = None
        best_below_gap: float = float("inf")

        for block in text_blocks:
            if not block or len(block) < 5:
                continue

            x0, y0, x1, y1, text = block[:5]

            if text is None:
                continue

            stripped_text = text.strip()
            if not stripped_text:
                continue

            bbox = BoundingBox(x0, y0, x1, y1)
            bbox_tuple = bbox.to_tuple()

            # Never reuse the same text block for multiple tables.
            if bbox_tuple in used_attached_bboxes:
                continue

            # Caption-like block only.
            if not is_small_text_block(stripped_text, max_lines=max_lines, max_chars=max_chars):
                continue

            # Require at least some horizontal alignment with the table.
            if horizontal_overlap_width(table_bbox, bbox) <= 0.0:
                continue

            # Candidate above the table.
            if is_close_above_table(
                table_bbox,
                bbox,
                max_distance_px=max_distance_px,
                max_overlap_px=max_overlap_px,
            ):
                gap = table_bbox.y0 - bbox.y1
                if abs(gap) < best_above_gap:
                    best_above_bbox = bbox_tuple
                    best_above_text = stripped_text
                    best_above_gap = abs(gap)

            # Candidate below the table.
            if is_close_below_table(
                table_bbox,
                bbox,
                max_distance_px=max_distance_px,
                max_overlap_px=max_overlap_px,
            ):
                gap = bbox.y0 - table_bbox.y1
                if abs(gap) < best_below_gap:
                    best_below_bbox = bbox_tuple
                    best_below_text = stripped_text
                    best_below_gap = abs(gap)

        # If the same tiny block somehow qualifies for both above and below,
        # prefer the above caption and do not duplicate it below.
        if best_above_bbox is not None and best_below_bbox == best_above_bbox:
            best_below_bbox = None
            best_below_text = None

        if best_above_bbox is not None and best_above_text is not None:
            table_data.headline = combine_nonempty_texts(
                best_above_text,
                table_data.headline,
            )
            used_attached_bboxes.add(best_above_bbox)

        if best_below_bbox is not None and best_below_text is not None:
            table_data.description = combine_nonempty_texts(
                table_data.description,
                best_below_text,
            )
            used_attached_bboxes.add(best_below_bbox)

        return table_data

