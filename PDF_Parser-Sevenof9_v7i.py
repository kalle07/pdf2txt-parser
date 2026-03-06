# Standard library
import os
import re
import sys
import time
import json
import platform
import threading
import logging
import subprocess
from typing import Any, Dict, Iterable, List, Sequence, Tuple, ClassVar
from dataclasses import dataclass, field, replace
import math

# Third-party libraries
import wx
import pdfplumber
import psutil
from pdfminer.pdfparser import PDFParser, PDFSyntaxError
from pdfminer.pdfdocument import PDFDocument, PDFEncryptionError, PDFPasswordIncorrect
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
import numpy as np

# Concurrency
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

# --------------------------------------------------------------
#   1. Configuration & compiled regexes
# --------------------------------------------------------------
PARALLEL_THRESHOLD = 16


@dataclass(frozen=True)
class Config:
    PARALLEL_THRESHOLD: int = 16          # pages per file before we switch to parallel mode
    
    # Class‑level constant – accessible via Config.TEXT_EXTRACT_SETTINGS
    TEXT_EXTRACT_SETTINGS: ClassVar[Dict[str, Any]] = {
        "x_tolerance": 1.5,
        "y_tolerance": 2.5,
        "keep_blank_chars": False,
        "use_text_flow": False,
    }
    
    LEFT_RIGHT_MARGIN_PCT: float = 5.3 # percent of the page
    TOP_BOTTOM_MARGIN_PCT: float = 6.0 # percent of the page



#CID_PATTERN = re.compile(r"\$cid:\d+$")  # Fixed: removed incorrect trailing $
CID_PATTERN = re.compile(r"\(cid:\d+\)")
# NON_Keyboard Pattern
NON_PRINTABLE_RE = re.compile(r"[\x00-\x1F\x7F\u200B-\u200D\uFEFF]")

def clean_cell_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove hyphenated line endings
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    # Remove CID patterns
    text = CID_PATTERN.sub("", text)
    # Remove non-printable/invisible characters
    text = NON_PRINTABLE_RE.sub("", text)
    return text.strip()


# --------------------------------------------------------------
#   2. Small utilities
# --------------------------------------------------------------

def get_physical_cores():
    count = psutil.cpu_count(logical=False)
    return max(1, count if count else 1)  # fallback = 1
cores = get_physical_cores()

# GUI update interval
def throttle_callback(callback, interval_ms=1):
    last_called = 0

    def wrapper(status):
        nonlocal last_called
        now = time.time() * 1000  # Time in ms
        if now - last_called >= interval_ms:
            last_called = now
            callback(status)
    return wrapper


def clamp_bbox(bbox: Tuple[float, float, float, float], w: float, h: float) -> Tuple[int, int, int, int]:
    """Clamp a bbox to the page dimensions and round to nearest integer."""
    x0, top, x1, bottom = bbox
    return (
        round(max(0, min(x0, w))),
        round(max(0, min(top, h))),
        round(min(x1, w)),
        round(min(bottom, h)),
    )


def is_valid_cell(cell: Any) -> bool:
    """Return True if a cell contains something meaningful."""
    return bool(str(cell).strip() and len(str(cell).strip()) > 1)



# Function to suppress PDFMiner logging, reducing verbosity
def suppress_pdfminer_logging():
    for logger_name in [
        "pdfminer",  # Various pdfminer modules to suppress logging from
        "pdfminer.pdfparser",
        "pdfminer.pdfdocument",
        "pdfminer.pdfpage",
        "pdfminer.converter",
        "pdfminer.layout",
        "pdfminer.cmapdb",
        "pdfminer.utils"
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)  # Set logging level to ERROR to suppress lower levels

suppress_pdfminer_logging()

class StatusTracker:
    def __init__(self, total_pages):
        self.start_time = time.time()
        self.total_pages = total_pages
        self.processed_pages = 0

    def update(self, n=1):
        self.processed_pages += n

    def get_status(self):
        elapsed = time.time() - self.start_time
        pages_per_sec = round(self.processed_pages / elapsed) if elapsed > 0 else 0
        remaining_pages = self.total_pages - self.processed_pages
        est_time = (remaining_pages / pages_per_sec) / 60 if pages_per_sec > 0 else float('inf')
        return {
            "processed_pages": self.processed_pages,
            "total_pages": self.total_pages,
            "pages_per_sec": pages_per_sec,
            "elapsed_time": round(elapsed / 60, 2),
            "est_time": round(est_time, 2)
        }

# --------------------------------------------------------------
#   3. Data models
# --------------------------------------------------------------

@dataclass(frozen=True)
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    font_name: str
    bold: bool


@dataclass
class Block:
    words: List[Word] = field(default_factory=list)

    def bbox(self) -> Tuple[float, float, float, float]:
        if not self.words:
            return 0.0, 0.0, 0.0, 0.0
        x0 = min(w.x0 for w in self.words)
        y0 = min(w.y0 for w in self.words)
        x1 = max(w.x1 for w in self.words)
        y1 = max(w.y1 for w in self.words)
        return (x0, y0, x1, y1)


@dataclass
class ImageInfo:
    bbox: Tuple[float, float, float, float]
    obj: Any  # raw image dictionary from pdfplumber


# --------------------------------------------------------------
#   4. Union‑Find clustering
# --------------------------------------------------------------

class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

def cluster_words(words: Sequence[Word], max_dx: int, max_dy: int) -> List[Block]:
    """Group words into blocks based on proximity using optimized neighbor search."""
    n = len(words)
    if n == 0:
        return []

    uf = _UnionFind(n)

    def is_neighbor(word1: Word, word2: Word) -> bool:
        dx = max(0.0, max(word1.x0 - word2.x1, word2.x0 - word1.x1))
        dy = max(0.0, max(word1.y0 - word2.y1, word2.y0 - word1.y0))
        return dx <= max_dx and dy <= max_dy

    # Track which words have already been processed (4 neighbors found)
    processed = [False] * n
    
    for i in range(n):
        if processed[i]:
            continue
            
        neighbor_count = 0
        neighbors_found = []
        
        # Check against ALL other words - the key optimization is to stop early
        for j in range(n):
            if i == j:
                continue
                
            word1, word2 = words[i], words[j]
            
            if is_neighbor(word1, word2):
                neighbors_found.append(j)
                neighbor_count += 1
                
                # Early stopping as per your requirements:
                # 1. If we have at least 2 neighbors, the word belongs to a text block
                # 2. If we already have 4 neighbors (max possible in 2D), stop processing this word
                if neighbor_count >= 1: 
                    # Union with all found neighbors so far
                    for k in neighbors_found:
                        uf.union(i, k)
                    
                    # Second early stop - no need to check further when 4 neighbors found
                    if neighbor_count >= 4:
                        processed[i] = True
                        break
                        
        # Continue processing other words even if current word had < 2 neighbors

    # Build clusters
    clusters: Dict[int, List[Word]] = {}
    for idx in range(n):
        root = uf.find(idx)
        clusters.setdefault(root, []).append(words[idx])

    # Return as list of Blocks
    return [Block(wlist) for wlist in clusters.values()]



# --------------------------------------------------------------
#   5. Character index (vectorised)
# --------------------------------------------------------------

@dataclass
class CharIndex:
    xs0: np.ndarray
    xs1: np.ndarray
    tops: np.ndarray
    bottoms: np.ndarray
    texts: List[str]
    fonts: List[str]
    sizes: np.ndarray

    @classmethod
    def build(cls, chars: Sequence[Dict[str, Any]]) -> "CharIndex":
        return cls(
            xs0=np.array([float(c["x0"]) for c in chars]),
            xs1=np.array([float(c["x1"]) for c in chars]),
            tops=np.array([float(c["top"]) for c in chars]),
            bottoms=np.array([float(c["bottom"]) for c in chars]),
            texts=[c.get("text", "") for c in chars],
            fonts=[c.get("fontname", "") for c in chars],
            sizes=np.array([float(c.get("size", 0)) for c in chars]),
        )

    def inside(self, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
        return (
            (self.xs0 >= x0)
            & (self.xs1 <= x1)
            & (self.tops >= y0)
            & (self.bottoms <= y1)
        )


# --------------------------------------------------------------
#   6. Core extraction helpers
# --------------------------------------------------------------

def _extract_tables(page: pdfplumber.page.Page) -> List[Tuple[str, Any]]:
    """Return a list of JSON strings representing tables."""
    suppress_pdfminer_logging()
    raw_tables = page.extract_tables(
        {"text_x_tolerance": Config.TEXT_EXTRACT_SETTINGS["x_tolerance"]}
    )
    jsons = []

    def has_valid_printable(cell: str) -> bool:
        """At least one alphanumeric character (no punctuation, no whitespace)."""
        return any(ch.isalnum() for ch in cell)

    for tbl in raw_tables:
        if not tbl:
            continue

        cleaned = [[clean_cell_text(c) for c in row] for row in tbl]

        rows = len(cleaned)
        cols = max(len(r) for r in cleaned) if cleaned else 0

        # Flatten all cells
        all_cells = [cell for row in cleaned for cell in row]

        # --- New validation rules ---
        if rows == 1 and cols == 1:
            # Single-cell table
            if not has_valid_printable(all_cells[0]):
                continue

        elif rows >= 1 and cols >= 1:
            # At least one row OR at least one column
            if not any(has_valid_printable(cell) for cell in all_cells):
                continue

        # --- Existing header logic with 1-row fix ---
        header = cleaned[0]

        if rows == 1:
            # Single-row table → preserve content
            table_dict = [dict(enumerate(header))]

        elif header[0].strip() == "":
            # corner-empty table
            col_headers = header[1:]
            row_headers = [row[0] for row in cleaned[1:]]
            data_rows = cleaned[1:]

            table_dict = {}
            for rh, row in zip(row_headers, data_rows):
                table_dict[rh] = dict(zip(col_headers, row[1:]))

        else:
            # normal header table
            headers = header
            data_rows = cleaned[1:]
            table_dict = [
                dict(zip(headers, row))
                for row in data_rows
                if len(row) == len(headers)
            ]


        jsons.append(json.dumps(table_dict, indent=1, ensure_ascii=False))

    return jsons



def _filter_words(
    words: List[Dict[str, Any]],
    tables_bboxes: List[Tuple[int, int, int, int]],
) -> List[Dict[str, Any]]:
    """Keep all words, but clean each word of non-printable characters and table overlaps."""
    filtered = []
    for w in words:
        x0, top = float(w["x0"]), float(w["top"])
        # Skip words that overlap a table
        if any(bx0 <= x0 <= bx2 and by0 <= top <= by3 for bx0, by0, bx2, by3 in tables_bboxes):
            continue
        # Clean the word in-place
        w["text"] = clean_cell_text(w["text"])
        filtered.append(w)
    return filtered


def _build_word_info(
    words: List[Dict[str, Any]],
    char_index: CharIndex,
) -> List[Word]:
    """Convert raw pdfplumber words into Word dataclass instances."""
    def is_bold(name: str) -> bool:
        n = name.lower()
        return "bold" in n or "bd" in n or "black" in n

    word_objs: List[Word] = []
    for w in words:
        x0, y0, x1, y1 = map(float, (w["x0"], w["top"], w["x1"], w["bottom"]))
        mask = char_index.inside(x0, x1, y0, y1)
        sizes = char_index.sizes[mask]
        fonts = [char_index.fonts[i] for i in np.nonzero(mask)[0]]
        bolds = [is_bold(f) for f in fonts]

        font_size = float(sizes.max()) if sizes.size else 0.0
        word_objs.append(
            Word(
                text=w["text"],
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                font_size=font_size,
                font_name=fonts[0] if fonts else "Unknown",
                bold=bool(bolds),
            )
        )
    return word_objs


def _group_blocks(
    words: List[Word],
    page_width: float,
    page_height: float,
) -> List[Block]:
    """Cluster words into logical blocks using Union-Find, cleaning text"""

    merged_words = words


    # thresholds in pixel – derived from percentages
    max_dx = int(round(page_width * 0.014))   # 1.51 %, ~9px
    max_dy = int(round(page_height * 0.012))  # 1.43 %, ~12px
    blocks = cluster_words(merged_words, max_dx, max_dy)
    
    # Filter out empty blocks and single-character printable blocks
    filtered_blocks = []
    for block in blocks:
        combined_text = " ".join(w.text for w in block.words)
        stripped_text = combined_text.strip()

        if stripped_text and len(stripped_text) > 1:
            printable_chars = ''.join(c for c in stripped_text if not c.isspace())
            if len(printable_chars) > 1:
                filtered_blocks.append(block)
    
    return filtered_blocks



# --------------------------------------------------------------
#   7. Page worker – orchestrator
# --------------------------------------------------------------

def process_batch_worker(args: Tuple[str, List[int]]) -> List[Tuple[int, str]]:
    """
    Process a LIST of pages in a single worker call.
    
    Args:
        args: A tuple containing (file_path, list_of_page_indices).
        
    Returns:
        A list of tuples: [(page_number, text_output), ...]
    """
    file_path, page_indices = args
    
    # Open the PDF once for this batch. 
    # This is much faster than opening/closing per page in a parallel loop.
    try:
        with pdfplumber.open(file_path) as pdf:
            results = []
            
            for p_idx in page_indices:
                if p_idx >= len(pdf.pages):
                    continue
                    
                # --- Reuse the logic from your original process_page_worker ---
                # We inline it here to keep it self-contained and avoid global dependency issues 
                # if this file is split later. If you prefer, you can call a refactored 
                # 'process_single_page' function inside this loop.
                
                page = pdf.pages[p_idx]
                w, h = page.width, page.height

                # 1. Crop margins
                margin_x = w * Config.LEFT_RIGHT_MARGIN_PCT / 100.0
                margin_y = h * Config.TOP_BOTTOM_MARGIN_PCT / 100.0
                cropped_page = page.crop((margin_x, margin_y, w - margin_x, h - margin_y))

                # 2. Extract Tables
                tables_json = _extract_tables(cropped_page)
                
                # 3. Extract Words & Filter
                table_bboxes = [clamp_bbox(t.bbox, w, h) for t in cropped_page.find_tables()]
                raw_words = cropped_page.extract_words(**Config.TEXT_EXTRACT_SETTINGS)
                filtered_raw = _filter_words(raw_words, table_bboxes)
                char_index = CharIndex.build(cropped_page.chars)

                words = _build_word_info(filtered_raw, char_index)
                
                # Calculate average font size for the whole page (needed for heuristics)
                avg_font_size = float(np.mean([w.font_size for w in words])) if words else 0.0

                # 4. Group Blocks
                blocks = _group_blocks(words, w, h)

                # 5. Sorting (Reading Order)
                def reading_score(block: Block) -> Tuple[float, float]:
                    x0, y0, x1, y1 = block.bbox()
                    height = y1 - y0
                    width = x1 - x0
                    area_log = math.log1p(width * height)
                    return (y0 * 0.6 + x0 * 0.4 - area_log * 0.05, y0)

                blocks.sort(key=reading_score)

                # 6. Separate Large/Small Blocks & Promote Overlaps
                large_blocks: List[Block] = []
                small_blocks: List[Block] = []

                for block in blocks:
                    x0, y0, x1, y1 = block.bbox()
                    area = (x1 - x0) * (y1 - y0)
                    if area < 2000:
                        small_blocks.append(block)
                    else:
                        large_blocks.append(block)

                # Promote small blocks overlapping large blocks
                remaining_small_blocks: List[Block] = []
                for sblk in small_blocks:
                    x0_s, y0_s, x1_s, y1_s = sblk.bbox()
                    re = 12
                    x0_e, y0_e = x0_s - re, y0_s - re
                    x1_e, y1_e = x1_s + re, y1_s + re

                    promoted = False
                    for lblk in large_blocks:
                        x0_l, y0_l, x1_l, y1_l = lblk.bbox()
                        if not (x1_e < x0_l or x1_l < x0_e or y1_e < y0_l or y1_l < y0_e):
                            large_blocks.append(sblk)
                            promoted = True
                            break
                    if not promoted:
                        remaining_small_blocks.append(sblk)
                small_blocks = remaining_small_blocks

                # 7. Helper: Merge hyphenated words across lines (Inlined for clarity)
                def merge_hyphenated_lines(lines_list):
                    merged = []
                    i = 0
                    while i < len(lines_list):
                        current = lines_list[i]
                        if (i + 1 < len(lines_list) and current and current[-1].text.endswith("-")):
                            next_line = lines_list[i + 1]
                            if next_line:
                                left = current[-1]
                                right = next_line[0]
                                merged_word = replace(
                                    left,
                                    text=left.text[:-1] + right.text,
                                    x1=right.x1,
                                    y1=right.y1,
                                )
                                current = current[:-1] + [merged_word]
                                next_line = next_line[1:]
                                merged.append(current)
                                if next_line:
                                    merged.append(next_line)
                                i += 2
                                continue
                        merged.append(current)
                        i += 1
                    return merged

                # 8. Assemble Output for THIS PAGE
                lines_output: List[str] = [f"\n\n--- Page {p_idx + 1} ---\n\n"]

                # Process Large Blocks (Main Text)
                for block in large_blocks:
                    y_tolerance = 1.5
                    lines_dict: Dict[int, List[Word]] = {}
                    
                    # Sort words into lines based on Y-coordinate
                    for w in sorted(block.words, key=lambda w: w.y0):
                        placed = False
                        for key in lines_dict:
                            if abs(w.y0 - key) <= y_tolerance:
                                lines_dict[key].append(w)
                                placed = True
                                break
                        if not placed:
                            lines_dict[w.y0] = [w]

                    # Convert dict → ordered list of lines (sorted by Y then X)
                    lines_list = [sorted(lw, key=lambda w: w.x0) for lw in sorted(lines_dict.values(), key=lambda lw: lw[0].y0)]
                    
                    # Merge hyphenated words
                    lines_list = merge_hyphenated_lines(lines_list)

                    # Combine text
                    combined_lines = [" ".join(w.text for w in line) for line in lines_list]
                    combined_text = " ".join(combined_lines)

                    # 9. Labeling Heuristics
                    chapter_hits = 0
                    important_hits = 0
                    for wobj in block.words:
                        if len(wobj.text) < 4 and not any(c.isalpha() for c in wobj.text):
                            continue
                        size_ratio = wobj.font_size / avg_font_size if avg_font_size else 0.0
                        if size_ratio >= 1.15:
                            chapter_hits += 1
                        elif wobj.bold and size_ratio >= 1.08:
                            important_hits += 1

                    label: str | None = None
                    hits = chapter_hits + important_hits
                    if hits > 1 or (hits == 1 and chapter_hits):
                        label = "CHAPTER" if chapter_hits else "IMPORTANT"

                    line_text = f"[{label}] {combined_text}" if label else combined_text
                    lines_output.append(line_text)
                    lines_output.append("")

                # Append Tables
                for idx, tbl_json in enumerate(tables_json, 1):
                    lines_output.append(f'"table {idx}":\n{tbl_json}')

                # Append Small Blocks (Snippets)
                if small_blocks:
                    lines_output.append("\n--- Small text snippets far away from large text blocks ---")
                    for i, blk in enumerate(small_blocks, 1):
                        y_tolerance = 1.5
                        local_dict: Dict[int, List[Word]] = {}
                        for w in sorted(blk.words, key=lambda w: w.y0):
                            placed = False
                            for key in local_dict:
                                if abs(w.y0 - key) <= y_tolerance:
                                    local_dict[key].append(w)
                                    placed = True
                                    break
                            if not placed:
                                local_dict[w.y0] = [w]

                        lines_list = [sorted(lw, key=lambda w: w.x0) for lw in sorted(local_dict.values(), key=lambda lw: lw[0].y0)]
                        lines_list = merge_hyphenated_lines(lines_list)
                        
                        for line in lines_list:
                            txt = " ".join(w.text for w in line)
                            lines_output.append(txt)

                results.append((p_idx, "\n".join(lines_output)))

            return results

    except Exception as exc:
        # If the whole batch fails (e.g., file corrupted), log all pages in batch as error
        err_msg = f"[ERROR] Batch failed for {file_path}: {exc.__class__.__name__}: {exc}"
        logging.exception(err_msg)
        
        # Return error for each page in the batch
        return [(p_idx, f"[ERROR] Page {p_idx + 1} processing failed: {exc}") for p_idx in page_indices]



def _generate_batches(total_pages: int, chunk_size: int = 4):
    """Generator that yields chunks of page indices."""
    for start in range(0, total_pages, chunk_size):
        # Create a slice. This is a list of integers [start, start+1, ...]
        end = min(start + chunk_size, total_pages)
        yield (start, list(range(start, end)))


# Processing part 
def run_serial_batched(path, page_number, tracker=None, progress_callback=None, stop_flag=None):
    """Serial processing using the same batch logic (for consistency)."""
    results = []
    
    # We process 4 pages at a time even in serial to keep the worker function consistent
    for _, page_list in _generate_batches(page_number, chunk_size=4):
        if stop_flag and stop_flag.is_set():
            break
            
        batch_args = (path, page_list)
        
        try:
            # Direct call instead of executor.submit
            batch_results = process_batch_worker(batch_args)
            
            for res in batch_results:
                results.append(res)
                if tracker is not None:
                    tracker.update()
                
                if progress_callback and tracker is not None:
                    report_status(tracker, progress_callback)
                    
        except Exception as exc:
            logging.exception(f"Serial batch failed: {exc}")
            
    return results



def run_parallel_batched(path: str, page_number: int, tracker=None, progress_callback=None, stop_flag=None):
    """
    Runs the extraction in parallel using BATCHES of pages.
    
    Args:
        path: Path to PDF file.
        page_number: Total number of pages to process (0-indexed count, e.g., 10 for 10 pages).
        
    Returns:
        List of results [(page_idx, text), ...]
    """
    # Create the list of arguments: (file_path, [list_of_page_indices])
    batch_args = [(path, page_list) for _, page_list in _generate_batches(page_number, chunk_size=8)]
    
    results_map = {i: None for i in range(page_number)} # Dictionary to map result back to index
    
    def update_progress_and_store(result_tuple):
        """Helper to store result and update tracker."""
        if stop_flag and stop_flag.is_set():
            return
            
        page_idx, text_content = result_tuple
        
        # Store the result
        results_map[page_idx] = (page_idx, text_content)
        
        # Update Tracker
        if tracker is not None:
            tracker.update()
            
        # Callback
        if progress_callback and tracker is not None:
            report_status(tracker, progress_callback)

    max_workers = min(len(batch_args), get_physical_cores())
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        futures = {executor.submit(process_batch_worker, arg): arg for arg in batch_args}
        
        for future in concurrent.futures.as_completed(futures):
            if stop_flag and stop_flag.is_set():
                break
            
            try:
                batch_results = future.result()
                # batch_results is a list of (page_idx, text) tuples
                for res in batch_results:
                    update_progress_and_store(res)
            except Exception as exc:
                logging.exception(f"Batch job failed: {exc}")

    # Convert dict to sorted list
    final_results = [results_map[i] for i in range(page_number)]
    return [r for r in final_results if r is not None]




def report_status(tracker, progress_callback=None):
    status = tracker.get_status()
    if progress_callback:
        progress_callback(status)
    else:
        print(f"[STATUS] {status['processed_pages']}/{status['total_pages']} Seiten "
              f"({status['pages_per_sec']:} Seiten/s, "
              f"Elapsed: {status['elapsed_time']} Sek.)"
              f"Est Time: {status['est_time']} Sek.)")


def save_pdf(path, page_number, tracker=None, parallel=False, progress_callback=None, stop_flag=None):
    """Wrapper that selects the correct runner based on 'parallel' flag."""
    
    if stop_flag and stop_flag.is_set():
        return 0

    # Use the new batched runners
    if parallel:
        results = run_parallel_batched(path, page_number, tracker, progress_callback, stop_flag)
    else:
        results = run_serial_batched(path, page_number, tracker, progress_callback, stop_flag)

    # Filter and Sort
    results = [r for r in results if r]  # Filter None (bei Stop)
    results.sort(key=lambda x: x[0])     # Sort by page number
    
    text_output = "\n".join(text for _, text in results)

    out_path = os.path.splitext(path)[0] + ".txt"
    with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(text_output)

    return page_number



def _process_single_pdf(path):
    suppress_pdfminer_logging()
    try:
        with open(path, "rb") as f:
            parser = PDFParser(f)
            document = PDFDocument(parser)

            if not document.is_extractable:
                raise PDFTextExtractionNotAllowed("Text-Extraktion nicht erlaubt")

            pages = list(PDFPage.create_pages(document))
            return (path, len(pages), None)

    except (PDFEncryptionError, PDFPasswordIncorrect) as e:
        return (path, 0, f"[ERROR] Datei passwortgeschützt: {path} ({type(e).__name__}: {e})\n")
    except PDFSyntaxError as e:
        return (path, 0, f"[ERROR] Ungültige PDF-Syntax: {path} ({type(e).__name__}: {e})\n")
    except PDFTextExtractionNotAllowed as e:
        return (path, 0, f"[ERROR] Text-Extraktion nicht erlaubt: {path} ({type(e).__name__}: {e})\n")
    except Exception as e:
        return (path, 0, f"[ERROR] Fehler bei Datei {path}: {type(e).__name__}: {e}\n")

def get_total_pages(pdf_files, error_callback=None, progress_callback=None):
    suppress_pdfminer_logging()
    total = 0
    page_info = []

    def handle_result(path, count, error):
        nonlocal total
        if error:
            if error_callback:
                error_callback(error)
            else:
                print(error, end="")
        else:
            page_info.append((path, count))
            total += count
            if progress_callback:
                progress_callback(total)  # Rückmeldung an GUI

    if len(pdf_files) > 16:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
            results = executor.map(_process_single_pdf, pdf_files)
            for path, count, error in results:
                handle_result(path, count, error)
    else:
        for path in pdf_files:
            path, count, error = _process_single_pdf(path)
            handle_result(path, count, error)

    return page_info, total




# -------------------- GUI --------------------
class FileManager(wx.Frame):
    def __init__(self, parent):
        super().__init__(parent, title="PDF Parser - Sevenof9_v7i", size=(1000, 800))
        self.files = []
        self.InitUI()
        self.stop_flag = threading.Event()

    def InitUI(self):
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        hbox_lbl1 = wx.BoxSizer(wx.HORIZONTAL)

        lbl1 = wx.StaticText(panel, label="PDF files: (with right mouse you can remove and open)")
        hbox_lbl1.Add(lbl1, flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=10)

        hbox_lbl1.AddStretchSpacer()  # <== schiebt den Button ganz nach rechts

        help_btn = wx.Button(panel, label="? HELP ?", size=(60, 25))
        help_btn.Bind(wx.EVT_BUTTON, self.ShowHelpText)
        hbox_lbl1.Add(help_btn, flag=wx.RIGHT, border=10)

        vbox.Add(hbox_lbl1, flag=wx.EXPAND | wx.TOP, border=10)


        self.listbox = wx.ListBox(panel, style=wx.LB_EXTENDED)
        self.listbox.Bind(wx.EVT_RIGHT_DOWN, self.OnRightClick)
        self.listbox.Bind(wx.EVT_LISTBOX, self.ShowText)
        vbox.Add(self.listbox, proportion=1, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)

        self.popup_menu = wx.Menu()
        self.popup_menu.Append(1, "Remove selected")
        self.popup_menu.Append(2, "Open in default PDF app")
        self.popup_menu.Append(3, "Copy File Location")
        self.popup_menu.Append(4, "Open File Location")
        self.Bind(wx.EVT_MENU, self.RemoveFile, id=1)
        self.Bind(wx.EVT_MENU, self.OpenPDF, id=2)
        self.Bind(wx.EVT_MENU, self.CopyFileLocation, id=3)
        self.Bind(wx.EVT_MENU, self.OpenFileLocation, id=4)


        btn_panel = wx.Panel(panel)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        for label, handler in [
            ("Add Folder", self.AddFolder),
            ("Select Files", self.AddFile),
            ("Remove Selected", self.RemoveFile),
            ("Remove All", self.RemoveAll),
            ("Stop Parser", self.StopParser),
            ("Start Parser", self.StartParser)
        ]:
            btn = wx.Button(btn_panel, label=label)
            btn.Bind(wx.EVT_BUTTON, handler)
            if label == "Stop Parser":
                btn.SetBackgroundColour(wx.Colour(255, 180, 180))  # light red
            elif label == "Start Parser":
                btn.SetBackgroundColour(wx.Colour(180, 255, 180))  # light green
                self.start_btn = btn  # <-- Referenz merken
            btn_sizer.Add(btn, proportion=1, flag=wx.ALL, border=5)
        btn_panel.SetSizer(btn_sizer)
        vbox.Add(btn_panel, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)


        lbl2 = wx.StaticText(panel, label="Text Frame: (choose PDF to see converted text)")
        vbox.Add(lbl2, flag=wx.LEFT, border=10)

        self.text_ctrl = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.ShowHelpText(None)
        vbox.Add(self.text_ctrl, proportion=1, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)

        # Statusanzeige
        stat_grid = wx.FlexGridSizer(1, 5, 5, 55)
        self.lbl_processed_pages = wx.StaticText(panel, label="Processed pages: 0")
        self.lbl_total_pages = wx.StaticText(panel, label="Total pages: 0")
        self.lbl_pages_per_sec = wx.StaticText(panel, label="Pages/sec: 0")
        self.lbl_est_time = wx.StaticText(panel, label="Estimated time (min): 0.00")
        self.lbl_elapsed_time = wx.StaticText(panel, label="Elapsed time: 0.00")
        
        for lbl in [self.lbl_processed_pages, self.lbl_total_pages, self.lbl_pages_per_sec, self.lbl_est_time, self.lbl_elapsed_time]:
            stat_grid.Add(lbl)
        vbox.Add(stat_grid, flag=wx.LEFT | wx.TOP, border=10)

        self.prog_ctrl = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        vbox.Add(self.prog_ctrl, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)

        panel.SetSizer(vbox)


    def ShowHelpText(self, event):
        help_text = (
            "	This is a small help\n\n"
            "	• PRE ALPHA version (for ever) •\n"
            "• The generated TXT file has the same name as the PDF file.\n"
            "• The TXT file is created in the same directory as the PDF.\n"
            "• Older TXT files will be overwritten without prompting.\n"
            "• When selecting a folder, subfolders are also selected.\n"
            "• You can remove or open the PDF file by right-clicking on it.\n"
            "• Once everything has been processed, you can see the result immediately when you click on a PDF file.\n"
            "If:\n"
            "[INFO] File completed: TEST.pdf (X pages)!\n"
            "[INFO] Processing completed\n"
            "-> This only means that all pages have been processed; it does not mean that the quality is good.\n"
            "-> If you cannot select and copy the text in the PDF and paste it into an editor, this program will not work.\n"
            "-> No diagrams, NO images, and NO OCR will be processed.\n"
            "• An attempt is made to reproduce the layout of the page in columns from left to right and in blocks from top to bottom.\n"
            "• An attempt is made to detect regular tables with lines; headers (top or top and left) are assigned to the cells and stored in JSON format in the text file.\n"
            "• Adds the label “Page X” at the beginning of every page (absolute number).\n"
            "• Adds the label “Chapter” for large font and/or “Important” for bold font.\n"
            "\n"
            "Stop function becomes effective only after the currently processed file.\n"
            "When processing large amounts of data, the following should be noted:\n"
            "1. all PDFs are opened once to determine the number of pages.\n"
            "2. all PDFs are chunked for fast multiprocessing, need some seconds for large amount of files.\n"
            "3. all small PDFs are processed in parallel.\n"
            "4. each large PDF is processed page-chunk by page-chuck in parallel.\n"
        )
        self.text_ctrl.SetValue(help_text)
        
        
    def AddFolder(self, event):
        dlg = wx.DirDialog(self, "Select Folder")
        if dlg.ShowModal() == wx.ID_OK:
            for root, _, files in os.walk(dlg.GetPath()):
                for f in files:
                    if f.lower().endswith(".pdf"):
                        path = os.path.normpath(os.path.join(root, f))
                        if path not in self.files:
                            self.files.append(path)
                            self.listbox.Append(path)
        dlg.Destroy()

    def AddFile(self, event):
        with wx.FileDialog(self, "Select PDF Files", wildcard="PDF files (*.pdf)|*.pdf",
                           style=wx.FD_OPEN | wx.FD_MULTIPLE) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                for path in dlg.GetPaths():
                    if path not in self.files:
                        self.files.append(path)
                        self.listbox.Append(path)

    def RemoveFile(self, event):
        for i in reversed(self.listbox.GetSelections()):
            self.listbox.Delete(i)
            del self.files[i]
        self.text_ctrl.Clear()

    def RemoveAll(self, event):
        self.listbox.Clear()
        self.files.clear()
        self.text_ctrl.Clear()

    def OpenPDF(self, event):
        i = self.listbox.GetSelections()
        if i:
            path = self.files[i[0]]
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.call(["open", path])
            else:
                subprocess.call(["xdg-open", path])
                
    def CopyFileLocation(self, event):
        sel = self.listbox.GetSelections()
        if sel:
            path = self.files[sel[0]]
            if wx.TheClipboard.Open():
                wx.TheClipboard.SetData(wx.TextDataObject(path))
                wx.TheClipboard.Close()

    def OpenFileLocation(self, event):
        sel = self.listbox.GetSelections()
        if sel:
            folder = os.path.dirname(self.files[sel[0]])
            if platform.system() == "Windows":
                subprocess.Popen(f'explorer "{folder}"')
            elif platform.system() == "Darwin":
                subprocess.call(["open", folder])
            else:
                subprocess.call(["xdg-open", folder])


    def OnRightClick(self, event):
        if self.listbox.GetSelections():
            self.PopupMenu(self.popup_menu, event.GetPosition())

    def StartParser(self, event):
        if not self.files:
            wx.MessageBox("Please select files first.", "Hinweis", wx.OK | wx.ICON_INFORMATION)
            wx.CallAfter(self.start_btn.Enable)  # <-- wieder aktivieren
            return


        self.start_btn.Disable()
        self.stop_flag.clear()
        self.prog_ctrl.Clear()

        def error_callback(msg):
            wx.CallAfter(self.AppendProg, msg)
        
        def update_total_pages_live(new_total):
            wx.CallAfter(self.lbl_total_pages.SetLabel, f"Total pages: {new_total}")


        page_info, total_pages = get_total_pages(
            self.files,
            error_callback=error_callback,
            progress_callback=update_total_pages_live
        )

        if total_pages == 0:
            self.AppendProg("[INFO] No pages found.\n")
            wx.CallAfter(self.start_btn.Enable)  # <-- wieder aktivieren
            return

        tracker = StatusTracker(total_pages)

        def gui_progress_callback(status):
            wx.CallAfter(self.lbl_processed_pages.SetLabel, f"Processed pages: {status['processed_pages']}")
            wx.CallAfter(self.lbl_total_pages.SetLabel, f"Total pages: {status['total_pages']}")
            wx.CallAfter(self.lbl_pages_per_sec.SetLabel, f"Pages/sec: {status['pages_per_sec']:}")
            wx.CallAfter(self.lbl_est_time.SetLabel, f"Estimated time (min): {status['est_time']:}")
            wx.CallAfter(self.lbl_elapsed_time.SetLabel, f"Elapsed time: {status['elapsed_time']}")

        throttled_gui_callback = throttle_callback(gui_progress_callback, 100)

        def background():
            small = [p for p in page_info if p[1] <= PARALLEL_THRESHOLD]
            large = [p for p in page_info if p[1] > PARALLEL_THRESHOLD]

            # Verarbeite kleine Dateien je in einem eigenen Prozess
            if small:
                max_workers = max(1, min(len(small), get_physical_cores()))
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {}
                    for path, count in small:
                        if self.stop_flag.is_set():
                            break
                        future = executor.submit(save_pdf, path, count, None, False, None)
                        futures[future] = (path, count)

                    for future in concurrent.futures.as_completed(futures):
                        if self.stop_flag.is_set():
                            break
                        path, count = futures[future]
                        try:
                            pages_processed = future.result()
                            tracker.update(pages_processed)
                            throttled_gui_callback(tracker.get_status())
                            wx.CallAfter(self.AppendProg, f"[INFO] File ready: {path} ({pages_processed} Seiten)\n")
                        except Exception as e:
                            wx.CallAfter(self.AppendProg, f"[ERROR] File {path}: {str(e)}\n")

            # Verarbeite große Dateien Seite für Seite parallel
            for path, count in large:
                if self.stop_flag.is_set():
                    break

                try:
                    pages_processed = save_pdf(
                        path,
                        count,
                        tracker,
                        parallel=True,
                        progress_callback=throttled_gui_callback,
                        stop_flag=self.stop_flag
                    )
                    if pages_processed:
                        wx.CallAfter(
                            self.AppendProg,
                            f"[INFO] File ready: {path} ({pages_processed} Seiten)\n"
                        )
                    else:
                        wx.CallAfter(
                            self.AppendProg,
                            f"[INFO] Stopped: {path}\n"
                        )
                except Exception as e:
                    wx.CallAfter(
                        self.AppendProg,
                        f"[ERROR] File {path}: {str(e)}\n"
                    )



            wx.CallAfter(self.AppendProg, "\n[INFO] Processing completed.\n")
            wx.CallAfter(self.start_btn.Enable)  # <-- wieder aktivieren
            self.stop_flag.clear()

        threading.Thread(target=background, daemon=True).start()


    def StopParser(self, event):
        self.stop_flag.set()
        self.AppendProg("[INFO] Processing Stopped...\n")

    
    def ShowText(self, event):
        sel = self.listbox.GetSelections()
        if not sel:
            return
        txt_path = os.path.splitext(self.files[sel[0]])[0] + ".txt"
        self.text_ctrl.Clear()
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                self.text_ctrl.SetValue(f.read())
        else:
            self.text_ctrl.SetValue("[No .txt file found]")

    def AppendProg(self, text):
        self.prog_ctrl.AppendText(text)


# -------------------- Einstiegspunkt --------------------
def main():
    if len(sys.argv) > 1:
        pdf_files = sys.argv[1:]
        page_info, total_pages = get_total_pages(pdf_files)
        tracker = StatusTracker(total_pages)

        def cli_callback(status):
            print(json.dumps(status))

        for path, count in page_info:
            save_pdf(path, count, tracker, parallel=(count > PARALLEL_THRESHOLD), progress_callback=cli_callback)
    else:
        app = wx.App(False)
        frame = FileManager(None)
        frame.Show()
        app.MainLoop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()



