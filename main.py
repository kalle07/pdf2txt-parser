"""PDF Parser — enhanced wxPython GUI application.

Converts PDFs to plain text, images, and drawings using a multiprocess pipeline.
Features:
    • Add PDFs or whole folders, right‑click to remove, double‑click to open
    • Initial scan counts pages of every PDF in the list
    • Live counters: PDFs left, pages left, pages/sec, ETA, images/drawings saved
    • Thread‑safe progress updates via wx.CallAfter
    • Per‑file progress tracking for accurate "files left" counter
    • After processing, select a PDF to view its converted .txt inline
"""

import logging
import json
import multiprocessing
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque
from threading import Thread, Event
from typing import Any, Callable, Dict, List, Optional

import wx
import wx.lib.newevent

from converter import (
    ConversionConfig,
    PDFConverter,
    PDFValidator,
    get_physical_cores,
)

# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("pdf_parser")

# --------------------------------------------------------------------------- #
#  Progress event (unused when wx.CallAfter is used, kept for compatibility)
# --------------------------------------------------------------------------- #
ProgressEvent, EVT_PROGRESS = wx.lib.newevent.NewEvent()





class PDFFolderDropTarget(wx.FileDropTarget):
    """Collects dropped files/folders, filters for .pdf only."""

    def __init__(self, window: wx.Window) -> None:
        super().__init__()
        self._window = window

    def _collect_pdfs(self, path: str) -> List[str]:
        lower = path.lower()

        if lower.endswith(".pdf"):
            return [path]

        if os.path.isdir(path):
            pdfs: List[str] = []

            for root, _, files in os.walk(path):
                for f in files:
                    if f.lower().endswith(".pdf"):
                        full = os.path.normpath(os.path.join(root, f))
                        pdfs.append(full)

            return pdfs

        return []

    def _set_highlight(self, enabled: bool) -> None:
        panel = self._window._file_list_panel
        file_list = self._window.file_list

        if enabled:
            highlight = wx.Colour(0xE8, 0xF4, 0xE8)

            panel.SetBackgroundColour(highlight)
            file_list.SetBackgroundColour(highlight)

        else:
            panel.SetBackgroundColour(
                self._window._theme.get(
                    "border_grey",
                    self._window._theme["panel_bg"],
                )
            )
            file_list.SetBackgroundColour(
                self._window._theme.get(
                    "list_bg",
                    self._window._theme["reader_bg"],
                )
            )

        panel.Refresh()
        file_list.Refresh()
        panel.Update()
        file_list.Update()

    def OnEnter(self, x: int, y: int, d: int) -> int:
        self._set_highlight(True)
        return d

    def OnDragOver(self, x: int, y: int, d: int) -> int:
        self._set_highlight(True)
        return d

    def OnLeave(self) -> None:
        self._set_highlight(False)

    def OnDropFiles(self, x: int, y: int, paths: List[str]) -> bool:
        self._set_highlight(False)

        all_pdfs: List[str] = []

        for path in paths:
            all_pdfs.extend(self._collect_pdfs(path))

        if all_pdfs:
            wx.CallAfter(self._window._add_to_list, all_pdfs)

        return True







# --------------------------------------------------------------------------- #
#  Per‑file scan info
# --------------------------------------------------------------------------- #
@dataclass
class FileInfo:
    """Holds metadata for a single PDF entry in the file list."""

    path: str = ""
    name: str = ""
    pages: int = 0
    status: str = "Scanning..."
    error: str = ""
    pages_done: int = 0          # pages processed so far (during conversion)

# --------------------------------------------------------------------------- #
#  Worker thread
# --------------------------------------------------------------------------- #
class ConverterThread(Thread):
    """Runs the conversion pipeline in a background thread."""

    def __init__(
        self,
        pdf_files: List[str],
        output_folder: str,
        cores: int,
        config: ConversionConfig,
        progress_callback: Optional[Callable[[], None]] = None,
        finish_callback: Optional[Callable[[], None]] = None,
        stop_flag: Optional[Event] = None,
    ):
        super().__init__(daemon=True)
        self.pdf_files = pdf_files
        self.output_folder = output_folder
        self.cores = cores
        self.config = config
        self.progress_callback = progress_callback
        self.finish_callback = finish_callback
        self.stop_flag = stop_flag
        self.global_stats: Dict[str, Any] = {}

    def run(self) -> None:
        start_time = time.time()
        converter = PDFConverter(
            self.config,
            progress_callback=self.progress_callback,
            stop_flag=self.stop_flag,
        )
        self.global_stats = converter.convert_parallel(
            self.pdf_files,
            self.output_folder,
            self.cores,
            output_in_source_dir=True,
        )
        elapsed = time.time() - start_time
        pages = self.global_stats.get("total_pages", 0)
        pps = pages / elapsed if elapsed > 0 else 0

        if self.finish_callback:
            self.finish_callback(
                ProgressEvent(
                    success=True,
                    elapsed=elapsed,
                    pages=pages,
                    pps=pps,
                    stats=self.global_stats,
                )
            )

# --------------------------------------------------------------------------- #
#  Post-process thread – fills descriptions from existing .txt + _media only
# --------------------------------------------------------------------------- #
class PostProcessThread(Thread):
    """Run standalone post-processing over a set of directories.

    No PDF is opened: it scans each directory for ``.txt`` files and fills
    image/drawing descriptions from their ``_media`` folders. Missing or locked
    files are skipped by the post-processor itself, so this never crashes.
    """

    def __init__(
        self,
        directories: List[str],
        finish_callback: Optional[Callable] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(daemon=True)
        self.directories = directories
        self.finish_callback = finish_callback
        self.log_callback = log_callback

    def run(self) -> None:
        # Imported here so a missing module can't break app startup
        from post_process import process_directory

        totals: Dict[str, int] = {}
        for directory in self.directories:
            if self.log_callback:
                self.log_callback(f"Post-processing: {directory}")
            stats = process_directory(directory)
            for key, value in stats.items():
                totals[key] = totals.get(key, 0) + int(value)

        if self.finish_callback:
            self.finish_callback(totals)


# --------------------------------------------------------------------------- #
#  Scan thread – counts pages of added PDFs without blocking the GUI
# --------------------------------------------------------------------------- #
class ScanThread(Thread):
    """Validate PDFs and fill in page counts."""

    def __init__(self, pdf_paths: List[str], callback: Callable):
        super().__init__(daemon=True)
        self.pdf_paths = pdf_paths
        self.callback = callback

    def run(self) -> None:
        for path in self.pdf_paths:
            result = PDFValidator.validate_file(path)
            self.callback(
                path=path,
                valid=result["valid"],
                pages=result.get("page_count", 0),
                reason=result.get("reason", ""),
            )




# --------------------------------------------------------------------------- #
#  Help / info text
# --------------------------------------------------------------------------- #
HELP_TEXT = """\
    PDF to TXT Converter — Layout Focus & Table/JSON & Media Extraction
    

• The generated TXT file has the same name as the PDF file.
• The TXT file and media folders with images/drawings/formulas are created in the same directory.
• Older TXT files will be overwritten without prompting.
• When selecting a folder, all .pdf files inside it (non-hidden) are processed.

Right-click options:
• You can remove or open the source/converted PDF by right-clicking on it.

Status indicators after processing:
If:
[INFO] File completed: TEST.pdf (X pages)!
[INFO] Processing completed
-> This only means all pages were processed; image/drawing/table quality is not guaranteed.
-> If you cannot select and copy the text from your PDF, this program will produce poor results.
-> No OCR or AI-based recognition — pure pymupdf extraction only.

Layout & Content Rules:
• An attempt is made to reproduce page layout in columns (left → right) and blocks (top → bottom).
• Not straight markdown format, so two columns are not side by side (visual) but one above the other -> this is essential for embedding.
• some common types of tables with detectable structure are extracted and stored as JSON inside the TXT file -> best for embedding.
• Only join words split by a hyphen at the end of a line -> best for embedding.
• Adds "Page X of Y" label at the beginning of every processed page.
• Experimental, only formlas made with ms-office are saved as image

Image/Drawings Extraction:
• Images below 100 px on any side are skipped by default (adjustable via config).
• Full-page images (≥80% of page size) are excluded — likely background/scan artifacts.
• Images overlapping >90% with a similarly-sized text block or table are skipped.
• Max 10 media items per page to prevent cluttered output.

Drawing Extraction:
• A "drawing" requires at least 10 drawing rectangles clustered together (only configurable via min_items_per_cluster in python-file).
• Small blocks of text near drawings are incorporated into the cluster for contextual reasons.
• Drawings are saved with a border around their bounding box to accommodate captions or figure labels.

Margin & Overlap Protection:
• Content whose center falls within outer margins is skipped (configurable thresholds per side only in python-file).
• Tables take precedence — text blocks and drawings overlapping a table area by >90% are discarded.
• Images vs Drawings conflict resolution keeps the larger item; smaller one is logged as skipped.

Post-Processing Mode:
• First: describe all images and drawings beeing saved oc with help of Ai (Suggestion: LFM2.5-VL-1.6B)
• This second pass reads existing text files with-in pdf media-folder same name as the PDF and injects a description field alongside each image/drawing JSON block.
example: testfile_page_0003_img_02.png -> testfile_page_0003_img_02.txt

Stop function becomes effective only after the currently processed file finishes its page chunk.

When processing large amounts of data, the following should be noted:
1. All PDFs are opened once by PDFValidator to determine validity, protection status, and page count.
2. Files with fewer than 32 pages run in parallel — one core per file (up to available cores).
3. Large files (≥32 pages) are split into chunks of ~8 pages per core.
4. Each page runs inside a separate ProcessPoolExecutor worker, fully isolated with its own ConversionConfig copy.
5. Results from all workers are collected and assembled in original page order before writing the final TXT file.
6. Speed: 8 cores  ~50 pages / sec

...
https://github.com/kalle07/pdf2txt-parser

"""




# --------------------------------------------------------------------------- #
#  Main frame
# --------------------------------------------------------------------------- #
class PDFParserFrame(wx.Frame):
    """Main application window."""

    # ---- status‑label colour constants (never change with theme) ----
    C_PENDING  = wx.Colour(0xC0, 0xC0, 0xC0)   # grey
    C_SCANNING = wx.Colour(0xFF, 0xD7, 0x00)   # amber
    C_READY    = wx.Colour(0x40, 0xA0, 0x40)   # green
    C_PROCESS  = wx.Colour(0x00, 0x60, 0xC0)   # blue
    C_DONE     = wx.Colour(0x20, 0x80, 0x20)   # dark green  (red/green label – preserved)
    C_ERROR    = wx.Colour(0xCC, 0x00, 0x00)   # red          (red/green label – preserved)
    C_FEW_CHARS = wx.Colour(0xFF, 0x80, 0x00)  # orange (warning)    

    # ---- theme presets ----
    THEME_LIGHT = {
        "name": "Windows 95",
        "bg":       wx.Colour(0xC0, 0xC0, 0xC0),  # classic Win95 desktop gray
        "panel_bg": wx.Colour(0xD8, 0xD8, 0xD8),  # panel face
        "border":   wx.Colour(0x80, 0x80, 0x80),  # classic border gray
        "text":     wx.Colour(0x00, 0x00, 0x00),  # black text
        "accent":   wx.Colour(0x00, 0x00, 0x80),  # titlebar blue
        "header_bg": wx.Colour(0x00, 0x00, 0x80),  # titlebar blue
        "header_fg": wx.Colour(0xFF, 0xFF, 0xFF),  # white text on titlebar
        "btn_bg":   wx.Colour(0xC0, 0xC0, 0xC0),  # Win95 button face
        "input_bg": wx.Colour(0xFF, 0xFF, 0xFF),  # white input surface
        "border_white": wx.Colour(0xFF, 0xFF, 0xFF),  # light border stroke
        "border_grey": wx.Colour(0x80, 0x80, 0x80),  # dark border stroke
        "reader_bg": wx.Colour(0xFF, 0xFF, 0xFF),  # shared reading area
        "list_bg":  wx.Colour(0xD8, 0xD8, 0xD8),  # white list
        "log_bg":   wx.Colour(0xFF, 0xFF, 0xFF),  # white log
        "status_bg": wx.Colour(0xD8, 0xD8, 0xD8),  # status panel
        "btn_start": wx.Colour(0xB4, 0xFF, 0xB4),  # light green
        "btn_cancel": wx.Colour(0xFF, 0xB4, 0xB4),  # light red
        "btn_text": wx.Colour(0x00, 0x00, 0x00),
    }

    THEME_DARK = {
        "name": "Windows 95 Dark",
        "bg":       wx.Colour(0x2A, 0x2A, 0x2A),  # dark win95 desktop gray
        "panel_bg": wx.Colour(0x35, 0x35, 0x35),  # darker panel gray
        "border":   wx.Colour(0x5A, 0x5A, 0x5A),  # stronger edge contrast
        "text":     wx.Colour(0xE8, 0xE8, 0xE8),  # bright text on dark
        "accent":   wx.Colour(0x66, 0x66, 0x66),  # muted gray accent
        "header_bg": wx.Colour(0x42, 0x42, 0x42),  # gray title bar
        "header_fg": wx.Colour(0xF0, 0xF0, 0xF0), # off‑white on header
        "btn_bg":   wx.Colour(0x4A, 0x4A, 0x4A),  # Win95 dark button face
        "input_bg": wx.Colour(0x28, 0x28, 0x28),  # dark input/read surface
        "border_white": wx.Colour(0xD0, 0xD0, 0xD0),  # light stroke
        "border_grey": wx.Colour(0x7A, 0x7A, 0x7A),  # mid gray stroke
        "reader_bg": wx.Colour(0x22, 0x22, 0x22),  # dark reading area
        "list_bg":  wx.Colour(0x1E, 0x1E, 0x1E),  # dark list area
        "log_bg":   wx.Colour(0x18, 0x18, 0x18),  # darker log
        "status_bg": wx.Colour(0x36, 0x36, 0x36),  # status panel
        "btn_start": wx.Colour(0x3D, 0x73, 0x3D),  # muted green
        "btn_cancel": wx.Colour(0x73, 0x3D, 0x3D),  # muted red
        "btn_text": wx.Colour(0xE8, 0xE8, 0xE8),  # soft white button labels
    }

    THEME_PRESETS = {
        "Light": THEME_LIGHT,
        "Dark": THEME_DARK,
    }
    THEME_NAME_ALIASES = {
        "Windows 95": "Light",
        "Windows 95 Dark": "Dark",
    }
    THEME_COLOUR_KEYS = (
        "bg",
        "panel_bg",
        "border",
        "text",
        "accent",
        "header_bg",
        "header_fg",
        "btn_bg",
        "input_bg",
        "border_white",
        "border_grey",
        "reader_bg",
        "list_bg",
        "log_bg",
        "status_bg",
        "btn_start",
        "btn_cancel",
        "btn_text",
    )
    THEME_SETTINGS_FILE = "theme_settings.json"

    # columns widths    
    FILE_COLUMN_MIN_WIDTHS = [160, 50, 80, 70, 60, 70, 60] # need to add new column size if necessary



    def __init__(self) -> None:
        super().__init__(
            parent=None,
            title="PDF Parser by Kalle07",
            size=(960, 720),
            style=wx.DEFAULT_FRAME_STYLE,
        )

        # ---- theme state ----
        self._theme: Dict[str, wx.Colour] = dict(self.THEME_LIGHT)
        self._theme_name: str = "Light"
        self._load_theme_settings()

        # ---- internal state ----
        self.file_info: Dict[int, FileInfo] = {}  # list index → FileInfo
        self.converter_thread: Optional[ConverterThread] = None
        self.post_process_thread: Optional[PostProcessThread] = None
        self._stop_flag = Event()  # thread-safe cancellation signal

        # live‑progress state (safe to read from wx.CallAfter because the
        # worker thread is the *only* writer)
        self._pages_done: int = 0
        self._last_progress_update: float = 0.0
        self._total_pages: int = 0
        self._files_done: int = 0
        self._total_files: int = 0
        self._images_saved: int = 0
        self._drawings_saved: int = 0
        self._formulas_saved: int = 0
        self._start_time: float = 0.0
        self._is_processing: bool = False

        # rolling window for 1‑second pages/sec mean
        self._pps_window: deque[tuple[float, int]] = deque()

        # per‑file page tracker (written by worker thread, read by GUI thread)
        self._per_file_pages_done: Dict[str, int] = {}
        self._per_file_images: Dict[str, int] = {}
        self._per_file_drawings: Dict[str, int] = {}
        self._per_file_formulas: Dict[str, int] = {}

        # running sums updated as scanning completes
        self._total_pages_all: int = 0
        self._total_files_all: int = 0

        self._col_widths: Dict[int, int] = {}

        # File-list sorting state.
        # First click uses the requested initial direction.
        self._sort_directions: Dict[int, bool] = {
            0: False,  # File: A-Z
            1: False,  # Pages: low-high
            3: False,  # Status: Ready -> Processing -> Done
            4: True,   # Images: high-low
            5: True,   # Drawings: high-low
            6: True,   # Formulas: high-low
        }
        self._sort_column: Optional[int] = None

        self._build_ui()
        self._build_context_menu()
        self.Bind(wx.EVT_SIZE, self._on_window_resize)
        if not self.file_info:
            self._show_help()   # show help text in the viewer on first launch        
        self.Center()

        # Let wx finish the initial layout, then fit all columns.
        wx.CallAfter(self._pin_file_list_right_edge)        

        # Drop target on the FRAME — catches drops even when app is in background
        self.SetDropTarget(PDFFolderDropTarget(self))

        self._column_drag_timer = wx.Timer(self)
        self._column_drag_column = -1

        self.Bind(wx.EVT_TIMER, self._on_column_drag_timer,
                  self._column_drag_timer)


    def _truncate_path_left(self, path: str, max_width: int) -> str:
        """Return a path shortened from the left, preserving the filename."""
        if not path:
            return ""

        dc = wx.ClientDC(self.file_list)
        dc.SetFont(self.file_list.GetFont())

        if dc.GetTextExtent(path)[0] <= max_width:
            return path

        prefix = "...\\"
        remaining = path

        # Remove characters from the left until the complete string fits.
        while remaining:
            candidate = prefix + remaining
            if dc.GetTextExtent(candidate)[0] <= max_width:
                return candidate
            remaining = remaining[1:]

        return prefix + Path(path).name


    def _on_window_resize(self, evt: wx.SizeEvent) -> None:
        if evt is not None:
            evt.Skip()
        if hasattr(self, "file_list"):
            wx.CallAfter(self._pin_file_list_right_edge)

    def _theme_settings_path(self) -> Path:
        base_dir = Path(wx.StandardPaths.Get().GetUserConfigDir()) / "pdf_parser"
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Unable to create theme settings directory '%s': %s", base_dir, exc)
        return base_dir / self.THEME_SETTINGS_FILE

    @staticmethod
    def _colour_to_hex(colour: wx.Colour) -> str:
        return f"#{colour.Red():02X}{colour.Green():02X}{colour.Blue():02X}"

    @staticmethod
    def _colour_from_value(value: Any, fallback: wx.Colour) -> wx.Colour:
        if isinstance(value, wx.Colour):
            return value
        if not isinstance(value, str):
            return fallback
        try:
            return wx.Colour(value)
        except Exception:
            return fallback

    def _theme_to_payload(self) -> Dict[str, Any]:
        colors: Dict[str, str] = {}
        for key in self.THEME_COLOUR_KEYS:
            colour = self._theme.get(key)
            if isinstance(colour, wx.Colour):
                colors[key] = self._colour_to_hex(colour)
            else:
                colors[key] = self._colour_to_hex(wx.Colour(0x00, 0x00, 0x00))

        return {
            "theme_name": self._theme_name,
            "colors": colors,
        }

    def _save_theme_settings(self) -> None:
        path = self._theme_settings_path()
        payload = self._theme_to_payload()
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Unable to save theme settings '%s': %s", path, exc)

    def _load_theme_settings(self) -> None:
        path = self._theme_settings_path()
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("Unable to read theme settings '%s': %s", path, exc)
            return

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid theme settings JSON '%s': %s", path, exc)
            return

        if not isinstance(payload, dict):
            logger.warning("Theme settings payload has invalid format in '%s'", path)
            return

        theme_name = payload.get("theme_name")
        if isinstance(theme_name, str):
            resolved_theme = self.THEME_NAME_ALIASES.get(theme_name, theme_name)
            if isinstance(resolved_theme, str) and resolved_theme in self.THEME_PRESETS:
                self._theme = dict(self.THEME_PRESETS[resolved_theme])
                self._theme_name = resolved_theme
            else:
                self._theme_name = theme_name

        colors = payload.get("colors")
        if isinstance(colors, dict):
            for key in self.THEME_COLOUR_KEYS:
                saved = colors.get(key)
                fallback = self._theme.get(key)
                if not isinstance(fallback, wx.Colour):
                    continue
                self._theme[key] = self._colour_from_value(saved, fallback)

    # ==================================================================== #
    #  UI construction
    # ==================================================================== #
    def _build_ui(self) -> None:
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self._build_toolbar(), 0, wx.ALL | wx.EXPAND, 4)

        file_panel = self._build_file_list_panel()
        file_panel.Bind(wx.EVT_SIZE, self._on_panel_resize)  # <-- ADD THIS
        sizer.Add(file_panel, 1, wx.ALL | wx.EXPAND, 4)

        sizer.Add(self._build_config_panel(), 0, wx.ALL | wx.EXPAND, 4)
        sizer.Add(self._build_status_panel(), 0, wx.ALL | wx.EXPAND, 4)
        sizer.Add(self._build_text_viewer(), 1, wx.ALL | wx.EXPAND, 4)
        sizer.Add(self._build_log_panel(), 0, wx.ALL | wx.EXPAND, 4)
        self.SetSizer(sizer)
        self.SetMinSize((640, 480))
        self._apply_theme()



    # ---- apply colours to every widget ----
    def _apply_theme(self) -> None:
        t = self._theme
        reader_bg = t.get("reader_bg", t["input_bg"])
        list_bg = t.get("list_bg", reader_bg)
        log_bg = t.get("log_bg", reader_bg)
        status_bg = t.get("status_bg", t.get("border_white", t["panel_bg"]))

        # frame background = base canvas and panel seams
        self.SetBackgroundColour(t["bg"])

        # toolbar panel
        self._toolbar_panel.SetBackgroundColour(t["panel_bg"])
        if hasattr(self, "_sep_toolbar"):
            self._sep_toolbar.SetBackgroundColour(t.get("border_grey", t["border"]))

        # file list panel
        self._file_list_panel.SetBackgroundColour(t.get("border_grey", t["panel_bg"]))
        self.file_list.SetBackgroundColour(list_bg)
        self.file_list.SetForegroundColour(t["text"])

        # config panel
        self._config_panel.SetBackgroundColour(t.get("border_grey", t["panel_bg"]))
        for lbl in (self._lbl_cores, self._lbl_min_size, self._lbl_max_items):
            lbl.SetForegroundColour(t["text"])
        for chk in (self.chk_images, self.chk_drawings, self.chk_formulas, self.chk_margin,
                     self.chk_hyphen, self.chk_metadata, self.chk_post_only):
            chk.SetForegroundColour(t["text"])
            chk.SetBackgroundColour(t["panel_bg"])
        for sp in (self.spin_cores, self.spin_min_size, self.spin_max_items):
            sp.SetBackgroundColour(t["input_bg"])
            sp.SetForegroundColour(t["text"])

        # status panel
        self._status_panel.SetBackgroundColour(status_bg)
        for lbl in (self.lbl_pdf_left, self.lbl_pages_left, self.lbl_pps,
                     self.lbl_eta, self.lbl_images, self.lbl_drawings, self.lbl_formulas):
            lbl.SetForegroundColour(t["text"])

        # text viewer
        self._text_viewer_panel.SetBackgroundColour(t.get("border_grey", t["panel_bg"]))
        self._lbl_text_viewer.SetForegroundColour(t["text"])
        self.txt_viewer.SetBackgroundColour(reader_bg)
        self.txt_viewer.SetForegroundColour(t["text"])

        # log panel
        self._log_panel.SetBackgroundColour(t.get("border_grey", t["panel_bg"]))
        self.lbl_log.SetForegroundColour(t["text"])
        self.log_ctrl.SetBackgroundColour(log_bg)
        self.log_ctrl.SetForegroundColour(t["text"])

        # themed action buttons
        for btn in (self.btn_add, self.btn_add_folder, self.btn_remove, self.btn_clear,
                     self.btn_theme_light, self.btn_theme_dark, self.btn_help):
            btn.SetBackgroundColour(t["btn_bg"])
            btn.SetForegroundColour(t.get("btn_text", t["text"]))
        self.btn_start.SetBackgroundColour(t.get("btn_start", t["btn_bg"]))
        self.btn_start.SetForegroundColour(t.get("btn_text", t["text"]))
        self.btn_cancel.SetBackgroundColour(t.get("btn_cancel", t["btn_bg"]))
        self.btn_cancel.SetForegroundColour(t.get("btn_text", t["text"]))

        # refresh
        self.Refresh()
        self.Update()

    # ---- toolbar: add file / add folder / remove / clear / settings ----
    def _build_toolbar(self) -> wx.Panel:
        panel = wx.Panel(self, style=wx.BORDER_SUNKEN)
        self._toolbar_panel = panel
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.btn_add = wx.Button(panel, label="Add PDFs")
        self.btn_add.Bind(wx.EVT_BUTTON, self._on_add)
        self.btn_add_folder = wx.Button(panel, label="Add Folder")
        self.btn_add_folder.Bind(wx.EVT_BUTTON, self._on_add_folder)
        self.btn_remove = wx.Button(panel, label="Remove")
        self.btn_remove.Bind(wx.EVT_BUTTON, self._on_remove)
        self.btn_clear = wx.Button(panel, label="Remove All")
        self.btn_clear.Bind(wx.EVT_BUTTON, self._on_clear)

        for btn in (self.btn_add, self.btn_add_folder, self.btn_remove, self.btn_clear):
            sizer.Add(btn, 0, wx.ALL, 1)

        # theme toggle buttons (Light / Dark)
        self.btn_theme_light = wx.Button(panel, label="☀ Light")
        self.btn_theme_light.Bind(wx.EVT_BUTTON, self._on_theme_light)
        self.btn_theme_dark = wx.Button(panel, label="🌙 Dark")
        self.btn_theme_dark.Bind(wx.EVT_BUTTON, self._on_theme_dark)
        for btn in (self.btn_theme_light, self.btn_theme_dark):
            sizer.Add(btn, 0, wx.ALL, 1)

        self.btn_help = wx.Button(panel, label="Help")
        self.btn_help.Bind(wx.EVT_BUTTON, self._on_help)
        sizer.AddStretchSpacer()   # grows → pushes Help to the right edge
        sizer.Add(self.btn_help, 0, wx.ALL, 1)

        panel.SetSizer(sizer)
        return panel

    def _on_theme_light(self, _evt: wx.CommandEvent) -> None:
        """Switch to the light theme."""
        self._theme = dict(self.THEME_LIGHT)
        self._theme_name = "Light"
        self._apply_theme()
        self._save_theme_settings()

    def _on_theme_dark(self, _evt: wx.CommandEvent) -> None:
        """Switch to the dark theme."""
        self._theme = dict(self.THEME_DARK)
        self._theme_name = "Dark"
        self._apply_theme()
        self._save_theme_settings()


    def _on_help(self, _evt: wx.CommandEvent) -> None:
        """Show/hide help info in the text viewer."""
        self._show_help()

    # ---- file list (report‑mode listctrl) ----
    def _build_file_list_panel(self) -> wx.Panel:
        panel = wx.Panel(self, style=wx.BORDER_SUNKEN)
        self._file_list_panel = panel
        sizer = wx.BoxSizer(wx.VERTICAL)

        self.file_list = wx.ListCtrl(
            panel,
            style=wx.LC_REPORT | wx.LC_HRULES | wx.LC_VRULES,
        )



        self.file_list.InsertColumn(0, "File", width=350)
        self.file_list.InsertColumn(1, "Pages", width=70)
        self.file_list.InsertColumn(2, "Progress", width=120)
        self.file_list.InsertColumn(3, "Status", width=100)
        self.file_list.InsertColumn(4, "Images", width=80)
        self.file_list.InsertColumn(5, "Drawings", width=90)
        self.file_list.InsertColumn(6, "Formulas", width=80)


        self._col_widths = {
            i: self.file_list.GetColumn(i).GetWidth()
            for i in range(7)
        }


        self.file_list.Bind(
            wx.EVT_LIST_COL_BEGIN_DRAG,
            self._on_file_column_begin_drag,
        )

        self.file_list.Bind(
            wx.EVT_LIST_COL_END_DRAG,
            self._on_file_column_end_drag,
        )

        self.file_list.Bind(
            wx.EVT_LIST_COL_CLICK,
            self._on_file_column_click,
        )

        # events
        self.file_list.Bind(wx.EVT_LIST_ITEM_RIGHT_CLICK, self._on_list_right_click)
        self.file_list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self._on_list_double_click)
        self.file_list.Bind(wx.EVT_LIST_ITEM_SELECTED, self._on_list_select)

        sizer.Add(self.file_list, 1, wx.EXPAND | wx.ALL, 1)
        panel.SetSizer(sizer)


        return panel



    # ------------------------------------------------------------------ #
    #  File-list sorting
    # ------------------------------------------------------------------ #
    def _on_file_column_click(self, evt: wx.ListEvent) -> None:
        """Sort the file list when a column header is clicked."""
        column = evt.GetColumn()

        # Progress is not a useful sorting column.
        if column == 2:
            return

        if column not in (0, 1, 3, 4, 5):
            return

        # Clicking the same column reverses the direction.
        # Clicking another column uses its configured initial direction.
        if self._sort_column == column:
            descending = not self._sort_directions[column]
            self._sort_directions[column] = descending
        else:
            descending = self._sort_directions[column]
            self._sort_column = column

        self._sort_file_list(column, descending)

    def _sort_file_list(self, column: int, descending: bool) -> None:
        """Sort FileInfo objects and rebuild the ListCtrl rows."""
        if not self.file_info:
            return

        items = list(self.file_info.values())

        if column == 0:
            # Alphabetical by filename, case-insensitive.
            key_func = lambda fi: fi.name.casefold()

        elif column == 1:
            # Numeric page count.
            key_func = lambda fi: fi.pages

        elif column == 3:
            # Requested order:
            # Ready -> Processing -> Done
            # Error/Scanning are placed after the normal states.
            status_order = {
                "Ready": 0,
                "Processing": 1,
                "Done": 2,
                "Error": 3,
                "Scanning...": 4,
            }
            key_func = lambda fi: status_order.get(fi.status, 99)

        elif column == 4:
            key_func = lambda fi: self._file_list_number(fi, 4)

        elif column == 5:
            key_func = lambda fi: self._file_list_number(fi, 5)

        else:
            return

        items.sort(key=key_func, reverse=descending)

        # Rebuild the dictionary so dictionary index == ListCtrl row.
        self.file_info = {
            idx: fi
            for idx, fi in enumerate(items)
        }

        # Rebuild the visible list.
        self.file_list.Freeze()
        try:
            self.file_list.DeleteAllItems()

            width = self.file_list.GetColumn(0).GetWidth() - 10

            for idx, fi in self.file_info.items():
                self.file_list.InsertItem(
                    idx,
                    self._truncate_path_left(fi.path, width),
                )

                self.file_list.SetItem(
                    idx,
                    1,
                    str(fi.pages) if fi.pages else "…",
                )

                if fi.status == "Error":
                    progress = "—"
                    status_text = fi.error or "Error"
                    colour = self.C_ERROR

                elif fi.status == "Scanning...":
                    progress = "…"
                    status_text = fi.status
                    colour = self.C_SCANNING

                elif fi.status == "Ready":
                    progress = f"{fi.pages} / 0"
                    status_text = fi.status
                    colour = self.C_READY

                elif fi.status == "Processing":
                    progress = f"{fi.pages - fi.pages_done} / {fi.pages_done}"
                    status_text = fi.status
                    colour = self.C_PROCESS

                elif fi.status == "Done":
                    progress = f"0 / {fi.pages}"
                    status_text = fi.status
                    colour = self.C_DONE

                else:
                    progress = "…"
                    status_text = fi.status
                    colour = self.C_PENDING

                self.file_list.SetItem(idx, 2, progress)
                self.file_list.SetItem(idx, 3, status_text)
                self.file_list.SetItem(
                    idx,
                    4,
                    self._file_list_media_value(fi, 4),
                )
                self.file_list.SetItem(
                    idx,
                    5,
                    self._file_list_media_value(fi, 5),
                )
                self.file_list.SetItemTextColour(idx, colour)

        finally:
            self.file_list.Thaw()

        self._refresh_file_paths()

    def _file_list_number(self, fi: FileInfo, column: int) -> int:
        """Return numeric value used for media sorting."""
        if column == 4:
            return self._per_file_images.get(fi.name, 0)

        if column == 5:
            return self._per_file_drawings.get(fi.name, 0)

        return 0

    def _file_list_media_value(self, fi: FileInfo, column: int) -> str:
        """Return current image/drawing count for a row."""
        return str(self._file_list_number(fi, column))



    def _on_column_drag_timer(self, _evt: wx.TimerEvent) -> None:
        """Continuously update the layout while a column divider is dragged."""
        if not hasattr(self, "file_list"):
            return

        # if self.file_list.GetColumnCount() != 6:
        if self.file_list.GetColumnCount() < 1:
            return

        column = self._column_drag_column
        if column < 0:
            return

        current = self.file_list.GetColumn(column).GetWidth()
        previous = self._col_widths.get(column, current)

        if current == previous:
            return


        # Apply the same balancing logic used by the final drag handler.
        self._apply_column_drag(column, current)


    def _on_file_column_begin_drag(self, evt: wx.ListEvent) -> None:
        """Start monitoring a native ListCtrl column resize."""
        self._column_drag_column = evt.GetColumn()

        # Store the widths immediately before native dragging starts.
        self._col_widths = {
            i: self.file_list.GetColumn(i).GetWidth()
            for i in range(self.file_list.GetColumnCount())
        }

        evt.Skip()

        if not self._column_drag_timer.IsRunning():
            self._column_drag_timer.Start(15)  # roughly 60–70 updates/sec


    def _apply_column_drag(self, column: int, new_width: int) -> None:
        count = self.file_list.GetColumnCount()
        min_widths = self.FILE_COLUMN_MIN_WIDTHS

        old_width = self._col_widths.get(column, new_width)
        delta = new_width - old_width

        if delta == 0:
            return

        if column == count - 1:
            # Keep Drawings' right edge fixed.
            drawings_width = old_width
            file_old = self._col_widths.get(
                0,
                self.file_list.GetColumn(0).GetWidth(),
            )
            file_new = max(min_widths[0], file_old + delta)

            self.file_list.SetColumnWidth(0, file_new)
            self.file_list.SetColumnWidth(column, drawings_width)

            self._col_widths[0] = file_new
            self._col_widths[column] = drawings_width

        else:
            neighbour = column + 1

            neighbour_old = self._col_widths.get(
                neighbour,
                self.file_list.GetColumn(neighbour).GetWidth(),
            )

            min_delta = min_widths[column] - old_width
            max_delta = neighbour_old - min_widths[neighbour]

            actual_delta = max(
                min_delta,
                min(delta, max_delta),
            )

            adjusted_width = old_width + actual_delta
            neighbour_new = neighbour_old - actual_delta

            self.file_list.SetColumnWidth(column, adjusted_width)
            self.file_list.SetColumnWidth(neighbour, neighbour_new)

            self._col_widths[column] = adjusted_width
            self._col_widths[neighbour] = neighbour_new

        self._refresh_file_paths()



    # --------------------------------------------------------------------------- #
    #  Excel-style column resizing with the last column pinned to the right
    # --------------------------------------------------------------------------- #
    def _on_file_column_end_drag(self, evt: wx.ListEvent) -> None:
        """Finish a column resize after live updates."""
        column = evt.GetColumn()

        if self._column_drag_timer.IsRunning():
            self._column_drag_timer.Stop()

        self._column_drag_column = -1

        # One final layout pass after the native drag has finished.
        wx.CallAfter(self._pin_file_list_right_edge)

        evt.Skip()

        count = self.file_list.GetColumnCount()
        #if count != 6 or not (0 <= column < count):
        if count < 1 or not (0 <= column < count):
            return

        min_widths = self.FILE_COLUMN_MIN_WIDTHS
        new_width = self.file_list.GetColumn(column).GetWidth()
        old_width = self._col_widths.get(column, new_width)
        delta = new_width - old_width
        if delta == 0:
            return

        # The outer/right edge of Drawings is fixed. A drag on that edge is
        # therefore cancelled; File absorbs the attempted width change.
        if column == count - 1:
            drawings_width = old_width
            file_old = self._col_widths.get(0, self.file_list.GetColumn(0).GetWidth())
            file_new = max(min_widths[0], file_old + delta)
            self.file_list.SetColumnWidth(0, file_new)
            self.file_list.SetColumnWidth(column, drawings_width)
            self._col_widths[0] = file_new
            self._col_widths[column] = drawings_width
            wx.CallAfter(self._pin_file_list_right_edge)
            return

        # Normal divider: the column to the right absorbs the inverse delta.
        neighbour = column + 1
        neighbour_old = self._col_widths.get(
            neighbour, self.file_list.GetColumn(neighbour).GetWidth()
        )
        min_delta = min_widths[column] - old_width
        max_delta = neighbour_old - min_widths[neighbour]

        actual_delta = max(
            min_delta,
            min(delta, max_delta),
        )
        new_width = old_width + actual_delta
        neighbour_new = neighbour_old - actual_delta

        self.file_list.SetColumnWidth(column, new_width)
        self.file_list.SetColumnWidth(neighbour, neighbour_new)
        self._col_widths[column] = new_width
        self._col_widths[neighbour] = neighbour_new
        wx.CallAfter(self._pin_file_list_right_edge)


    def _pin_file_list_right_edge(self) -> None:
        """Fit all ListCtrl columns inside the available width.

        Column 0 absorbs all remaining space. Other columns retain their
        current widths where possible, but are reduced proportionally if
        necessary to prevent horizontal scrolling.
        """
        if not hasattr(self, "file_list"):
            return

        count = self.file_list.GetColumnCount()
        if count < 1:
            return

        client_width = self.file_list.GetClientSize().GetWidth()
        if client_width <= 0:
            return

        min_widths = self.FILE_COLUMN_MIN_WIDTHS[:count]

        # Small safety allowance for the ListCtrl border/scrollbar.
        available = max(0, client_width - 2)

        widths = [
            self.file_list.GetColumn(i).GetWidth()
            for i in range(count)
        ]

        total_width = sum(widths)

        # ---------------------------------------------------------------
        # Case 1: Everything already fits.
        # Give all remaining space to File (column 0).
        # ---------------------------------------------------------------
        if total_width <= available:
            extra = available - total_width
            widths[0] += extra

        # ---------------------------------------------------------------
        # Case 2: Columns are too wide.
        # First shrink columns 1..N until everything fits.
        # Column 0 is kept at its minimum.
        # ---------------------------------------------------------------
        else:
            deficit = total_width - available

            # Shrink the non-file columns first, starting from the right.
            for i in range(count - 1, 0, -1):
                reducible = max(0, widths[i] - min_widths[i])
                take = min(deficit, reducible)

                widths[i] -= take
                deficit -= take

                if deficit <= 0:
                    break

            # If there is still a deficit, shrink File as a last resort.
            if deficit > 0:
                reducible = max(0, widths[0] - min_widths[0])
                take = min(deficit, reducible)
                widths[0] -= take
                deficit -= take

        # Apply widths.
        for i, width in enumerate(widths):
            width = max(min_widths[i], int(width))
            self.file_list.SetColumnWidth(i, width)
            self._col_widths[i] = width

        self._refresh_file_paths()


    def _resize_file_column(self) -> None:
        self._pin_file_list_right_edge()

    def _on_panel_resize(self, evt: wx.SizeEvent) -> None:
        evt.Skip()
        wx.CallAfter(self._pin_file_list_right_edge)


    # ------------------------------------------------------------------
    #  Drag‑and‑drop visual feedback (file list panel only)
    # ------------------------------------------------------------------
    def _on_drag_enter_file_list(self, evt: wx.MouseEvent) -> None:
        self._file_list_panel.SetBackgroundColour(wx.Colour(0xE8, 0xF4, 0xE8))
        self._file_list_panel.Refresh()
        evt.Skip()

    def _on_drag_leave_file_list(self, evt: wx.MouseEvent) -> None:
        self._file_list_panel.SetBackgroundColour(self._theme.get("border_grey", self._theme["panel_bg"]))
        self._file_list_panel.Refresh()
        evt.Skip()


    def _refresh_file_paths(self) -> None:
        """Refresh displayed paths using left-side truncation."""
        if not hasattr(self, "file_list"):
            return

        width = self.file_list.GetColumn(0).GetWidth() - 10

        for idx, fi in self.file_info.items():
            if idx < self.file_list.GetItemCount():
                self.file_list.SetItem(
                    idx,
                    0,
                    self._truncate_path_left(fi.path, width),
                )



    # ---- config panel (checkboxes + cores + start/stop) ----
    def _build_config_panel(self) -> wx.Panel:
        panel = wx.Panel(self, style=wx.BORDER_SUNKEN)
        self._config_panel = panel
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        # -------- left column: checkboxes + image options --------
        left_col = wx.BoxSizer(wx.VERTICAL)

        cb_row1 = wx.BoxSizer(wx.HORIZONTAL)

        self.chk_images = wx.CheckBox(panel, label="Save images")
        self.chk_images.SetValue(False)
        self.chk_images.SetToolTip(
            "Extract and save raster images found in the PDF."
        )

        self.chk_drawings = wx.CheckBox(panel, label="Save drawings")
        self.chk_drawings.SetValue(False)
        self.chk_drawings.SetToolTip(
            "Detect and save vector drawings found in the PDF."
        )

        self.chk_formulas = wx.CheckBox(panel, label="Save formulas")
        self.chk_formulas.SetValue(False)
        self.chk_formulas.SetToolTip(
            "Extract the Cambria Maths font/formula if the PDF file was saved from an Office .docx document."
        )

        self.chk_margin = wx.CheckBox(panel, label="Enable margin check")
        self.chk_margin.SetValue(True)
        self.chk_margin.SetToolTip(
            "Ignores ~5% of the page margins"
        )

        for chk in (self.chk_images, self.chk_drawings, self.chk_formulas, self.chk_margin):
            cb_row1.Add(chk, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)

        left_col.Add(cb_row1, 0, wx.EXPAND | wx.BOTTOM, 4)

        # -------- second row --------
        cb_row2 = wx.BoxSizer(wx.HORIZONTAL)

        self.chk_hyphen = wx.CheckBox(panel, label="Fix hyphenated words")
        self.chk_hyphen.SetValue(True)
        self.chk_hyphen.SetToolTip(
            "Join words split by a hyphen at the end of a line."
        )

        self.chk_metadata = wx.CheckBox(panel, label="Include metadata")
        self.chk_metadata.SetValue(True)
        self.chk_metadata.SetToolTip(
            "Include PDF metadata such as title, author and document information."
        )

        self.chk_post_only = wx.CheckBox(
            panel,
            label="Post-process only (no PDF)",
        )
        self.chk_post_only.SetValue(False)
        # Post-process-only mode: fill image/drawing descriptions from the
        # existing .txt + _media folders. The source PDF is not opened, so this
        # works on a folder of already-converted output with no re-conversion.        
        self.chk_post_only.SetToolTip(
            "Process existing TXT and _media folders only. "
            "The source PDF is not opened or converted."
        )

        for chk in (
            self.chk_hyphen,
            self.chk_metadata,
            self.chk_post_only,
        ):
            cb_row2.Add(chk, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)

        left_col.Add(cb_row2, 0, wx.EXPAND | wx.BOTTOM, 4)

        # -------- image options --------
        opts_row = wx.BoxSizer(wx.HORIZONTAL)

        self._lbl_min_size = wx.StaticText(
            panel,
            label="Image/Drawing min size px:",
        )
        self._lbl_min_size.SetToolTip(
            "Minimum width and height of an image/drawing to be saved. "
            "Images smaller than this size are skipped."
        )

        opts_row.Add(
            self._lbl_min_size,
            0,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
            4,
        )

        self.spin_min_size = wx.SpinCtrl(
            panel,
            value="100",
            min=10,
            max=5000,
        )
        self.spin_min_size.SetMinSize((68, -1))
        self.spin_min_size.SetToolTip(
            "Minimum image/drawing size in pixels. "
            "Example: 100 skips images smaller than 100 px on either side."
        )
        opts_row.Add(
            self.spin_min_size,
            0,
            wx.RIGHT,
            12,
        )

        self._lbl_max_items = wx.StaticText(
            panel,
            label="Max images(drawings)/page:",
        )
        self._lbl_max_items.SetToolTip(
            "Maximum number of images saved from a single PDF page."
        )

        opts_row.Add(
            self._lbl_max_items,
            0,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
            4,
        )

        self.spin_max_items = wx.SpinCtrl(
            panel,
            value="10",
            min=1,
            max=100,
        )
        self.spin_max_items.SetMinSize((68, -1))
        self.spin_max_items.SetToolTip(
            "Maximum number of image/media items that can be saved from one page."
        )

        opts_row.Add(
            self.spin_max_items,
            0,
            wx.ALL,
            2,
        )

        left_col.Add(opts_row, 0, wx.EXPAND)
        sizer.Add(left_col, 1, wx.ALL | wx.EXPAND, 4)   # proportional so it takes remaining space

        # -------- right column: cores + start/cancel (pushed to the edge via stretch spacer) --------
        row2 = wx.BoxSizer(wx.HORIZONTAL)
        self._lbl_cores = wx.StaticText(panel, label="Cores:")
        row2.Add(self._lbl_cores, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.spin_cores = wx.SpinCtrl(panel, min=1, max=64, initial=get_physical_cores())
        row2.Add(self.spin_cores, 0, wx.ALL, 2)

        # gui_example pattern: color Start green, Stop red
        self.btn_start = wx.Button(panel, label="Start Conversion")
        self.btn_start.Bind(wx.EVT_BUTTON, self._on_start)
        self.btn_start.SetBackgroundColour(self._theme.get("btn_start", wx.Colour(180, 255, 180)))
        self.btn_cancel = wx.Button(panel, label="Cancel")
        self.btn_cancel.Bind(wx.EVT_BUTTON, self._on_cancel)
        self.btn_cancel.SetBackgroundColour(self._theme.get("btn_cancel", wx.Colour(255, 180, 180)))
        self.btn_cancel.Enable(False)
        button_text = self._theme.get("btn_text", self._theme["text"])
        self.btn_start.SetForegroundColour(button_text)
        self.btn_cancel.SetForegroundColour(button_text)

        # ~20 % larger than the default Win95-style button size (~100×30)
        self.btn_start.SetMinSize((120, 36))
        self.btn_cancel.SetMinSize((90, 36))

        for btn in (self.btn_start, self.btn_cancel):
            row2.Add(btn, 0, wx.ALL, 2)

        sizer.AddStretchSpacer()   # grows → pushes everything after it to the right edge
        sizer.Add(row2, 0, wx.ALL, 4)
        panel.SetSizer(sizer)
        return panel

    # ---- status / progress panel ----
    def _build_status_panel(self) -> wx.Panel:
        panel = wx.Panel(self, style=wx.BORDER_SUNKEN)
        self._status_panel = panel
        sizer = wx.BoxSizer(wx.VERTICAL)

        # info line – two rows for readability
        info_row1 = wx.BoxSizer(wx.HORIZONTAL)
        self.lbl_pdf_left = wx.StaticText(panel, label="PDFs left: 0")
        self.lbl_pages_left = wx.StaticText(panel, label="Pages left: 0")
        self.lbl_pps = wx.StaticText(panel, label="Pages/sec: 0.0")
        self.lbl_eta = wx.StaticText(panel, label="ETA: --")
        for lbl in (self.lbl_pdf_left, self.lbl_pages_left, self.lbl_pps, self.lbl_eta):
            info_row1.Add(lbl, 1, wx.ALL | wx.ALIGN_CENTER, 2)
        sizer.Add(info_row1, 0, wx.EXPAND)

        info_row2 = wx.BoxSizer(wx.HORIZONTAL)
        self.lbl_images = wx.StaticText(panel, label="Images saved: 0")
        self.lbl_drawings = wx.StaticText(panel, label="Drawings saved: 0")
        self.lbl_formulas = wx.StaticText(panel, label="Formulas saved: 0")
        for lbl in (self.lbl_images, self.lbl_drawings, self.lbl_formulas):
            info_row2.Add(lbl, 1, wx.ALL | wx.ALIGN_CENTER, 2)
        sizer.Add(info_row2, 0, wx.EXPAND)

        # progress bar
        self.progress = wx.Gauge(panel, style=wx.GA_HORIZONTAL)
        sizer.Add(self.progress, 0, wx.EXPAND | wx.BOTTOM, 4)

        panel.SetSizer(sizer)
        return panel

    # ---- text viewer ----
    def _build_text_viewer(self) -> wx.Panel:
        panel = wx.Panel(self, style=wx.BORDER_SUNKEN)
        self._text_viewer_panel = panel
        sizer = wx.BoxSizer(wx.VERTICAL)

        lbl = wx.StaticText(panel, label="Converted Text Viewer")
        self._lbl_text_viewer = lbl
        sizer.Add(lbl, 0, wx.BOTTOM, 4)

        self.txt_viewer = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2,
        )
        sizer.Add(self.txt_viewer, 1, wx.EXPAND | wx.ALL, 1)

        panel.SetSizer(sizer)
        return panel

    # ---- log panel ----
    def _build_log_panel(self) -> wx.Panel:
        panel = wx.Panel(self, style=wx.BORDER_SUNKEN)
        self._log_panel = panel

        # Minimum/initial height of the log area.
        panel.SetMinSize((-1, 100))

        sizer = wx.BoxSizer(wx.VERTICAL)

        self.lbl_log = wx.StaticText(panel, label="Log")
        sizer.Add(self.lbl_log, 0, wx.BOTTOM, 2)

        self.log_ctrl = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY,
        )
        sizer.Add(self.log_ctrl, 1, wx.EXPAND | wx.ALL, 1)

        panel.SetSizer(sizer)
        return panel

    # ---- context menu (4 items, matching gui_example) ----
    def _build_context_menu(self) -> None:
        self.ctx_menu = wx.Menu()
        self.ctx_menu.Append(1, "Remove selected")
        self.ctx_menu.Append(2, "Open in default PDF app")
        self.ctx_menu.Append(3, "Copy File Location")
        self.ctx_menu.Append(4, "Open File Location")
        self.Bind(wx.EVT_MENU, self._on_ctx_remove, id=1)
        self.Bind(wx.EVT_MENU, self._on_ctx_open, id=2)
        self.Bind(wx.EVT_MENU, self._on_ctx_copy_location, id=3)
        self.Bind(wx.EVT_MENU, self._on_ctx_open_location, id=4)

    # ==================================================================== #
    #  File management
    # ==================================================================== #
    def _add_to_list(self, paths: List[str]) -> None:
        """Add PDF paths to the list and start background scan."""
        new_paths: List[str] = []
        for p in paths:
            if not p.lower().endswith(".pdf"):
                continue
            if p in [self.file_info[i].path for i in self.file_info]:
                continue  # duplicate
            idx = self.file_list.GetItemCount()
            fi = FileInfo(path=p, name=Path(p).stem)  # stem matches converter's pdf_filename
            self.file_info[idx] = fi

            self.file_list.InsertItem(
                idx,
                self._truncate_path_left(
                    fi.path,
                    self.file_list.GetColumn(0).GetWidth() - 10,
                ),
            )

            self.file_list.SetItem(idx, 1, "…")
            self.file_list.SetItem(idx, 2, "…")
            self.file_list.SetItem(idx, 3, fi.status)
            self.file_list.SetItemTextColour(idx, self.C_SCANNING)
            new_paths.append(p)

        if new_paths:
            ScanThread(new_paths, callback=self._on_scan_result).start()

    def _on_add(self, _evt: wx.CommandEvent) -> None:
        with wx.FileDialog(
            self,
            "Select PDFs",
            wildcard="PDF files (*.pdf)|*.pdf",
            style=wx.FD_MULTIPLE | wx.FD_OPEN,
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self._add_to_list(dlg.GetPaths())

    def _on_add_folder(self, _evt: wx.CommandEvent) -> None:
        # gui_example pattern: os.walk + normpath for explicit filtering
        with wx.DirDialog(self, "Choose folder") as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                folder = dlg.GetPath()
                pdfs = []
                for root, _, files in os.walk(folder):
                    for f in files:
                        if f.lower().endswith(".pdf"):
                            path = os.path.normpath(os.path.join(root, f))
                            if path not in [self.file_info[i].path for i in self.file_info]:
                                pdfs.append(path)
                self._add_to_list(pdfs)

    def _remove_selected_rows(self) -> None:
        """Delete selected rows and re-index ``file_info`` so dict keys
        always match the ListCtrl row positions after deletion."""
        selected = []
        idx = self.file_list.GetFirstSelected()
        while idx != -1:
            selected.append(idx)
            idx = self.file_list.GetNextSelected(idx)
        if not selected:
            return
        for idx in reversed(selected):
            self.file_list.DeleteItem(idx)
            self.file_info.pop(idx, None)
        # compact remaining entries: row N in the list == key N in the dict
        self.file_info = {
            new_idx: fi
            for new_idx, (_, fi) in enumerate(sorted(self.file_info.items()))
        }
        self.txt_viewer.Clear()

    def _on_remove(self, _evt: wx.CommandEvent) -> None:
        self._remove_selected_rows()

    def _on_clear(self, _evt: wx.CommandEvent) -> None:
        self.file_list.DeleteAllItems()
        self.file_info.clear()

    def _on_list_right_click(self, evt: wx.ListEvent) -> None:
        # wx.ListEvent uses GetIndex(), not m_itemIndex (Qt convention)
        # gui_example pattern: pass mouse position to PopupMenu
        self.file_list.SetFocus()
        self.file_list.Select(evt.GetIndex())
        self.PopupMenu(self.ctx_menu, evt.GetPoint())
        evt.Skip()

    def _on_ctx_remove(self, _evt: wx.CommandEvent) -> None:
        self._remove_selected_rows()

    def _on_ctx_open(self, _evt: wx.CommandEvent) -> None:
        # wx.ListCtrl pattern: GetFirstSelected + GetNextSelected
        idx = self.file_list.GetFirstSelected()
        if idx == -1:
            return
        fi = self.file_info.get(idx)
        if fi and fi.path:
            path = fi.path
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.call(["open", path])
            else:
                subprocess.call(["xdg-open", path])

    def _on_ctx_copy_location(self, _evt: wx.CommandEvent) -> None:
        # wx.ListCtrl pattern: GetFirstSelected + GetNextSelected
        idx = self.file_list.GetFirstSelected()
        if idx == -1:
            return
        fi = self.file_info.get(idx)
        if fi and fi.path:
            if wx.TheClipboard.Open():
                wx.TheClipboard.SetData(wx.TextDataObject(fi.path))
                wx.TheClipboard.Close()

    def _on_ctx_open_location(self, _evt: wx.CommandEvent) -> None:
        # wx.ListCtrl pattern: GetFirstSelected + GetNextSelected
        idx = self.file_list.GetFirstSelected()
        if idx == -1:
            return
        fi = self.file_info.get(idx)
        if fi and fi.path:
            folder = os.path.dirname(fi.path)
            if platform.system() == "Windows":
                subprocess.Popen(f'explorer "{folder}"')
            elif platform.system() == "Darwin":
                subprocess.call(["open", folder])
            else:
                subprocess.call(["xdg-open", folder])

    def _on_list_double_click(self, evt: wx.ListEvent) -> None:
        idx = evt.GetIndex()
        fi = self.file_info.get(idx)
        if fi and fi.path:
            os.startfile(fi.path)

    def _on_list_select(self, evt: wx.ListEvent) -> None:
        """Load the converted .txt for the selected PDF (if done)."""
        idx = evt.GetIndex()
        fi = self.file_info.get(idx)
        if not fi:
            return

        # gui_example pattern: os.path.splitext + errors="ignore"
        txt_path = os.path.splitext(fi.path)[0] + ".txt"
        self.txt_viewer.Clear()

        # Show validation info for invalid/protected files
        if fi.status == "Error":
            result = PDFValidator.validate_file(fi.path)
            lines = [
                f"PDF: {fi.name}",
                f"Valid: {result['valid']}",
                f"Protected: {result.get('is_protected', False)}",
                f"Repaired: {result.get('is_repaired', False)}",
                f"Pages: {result.get('page_count', 'N/A')}",
                f"Reason: {result.get('reason', fi.error)}",
            ]
            self.txt_viewer.SetValue("\n".join(lines))
            return

        if fi.status != "Done" or not os.path.exists(txt_path):
            self.txt_viewer.SetValue(f"({fi.name}) not converted yet")
            return

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            self.txt_viewer.SetValue(f.read())

    # ---- scan callback ----
    def _on_scan_result(self, path: str, valid: bool, pages: int, reason: str) -> None:
        """Called from the scan thread – schedules a GUI update via wx.CallAfter."""
        wx.CallAfter(
            self._update_scan_result,
            path=path,
            valid=valid,
            pages=pages,
            reason=reason,
        )

    def _update_scan_result(
        self, path: str, valid: bool, pages: int, reason: str
    ) -> None:
        """GUI‑thread callback applied after a PDF has been validated."""
        for idx, fi in self.file_info.items():
            if fi.path == path:
                fi.pages = pages if valid else 0
                fi.error = reason if not valid else ""

                if valid:
                    # Check if .txt file already exists from a previous conversion
                    txt_path = os.path.splitext(fi.path)[0] + ".txt"
                    if os.path.exists(txt_path):
                        fi.status = "Done"
                        # 0 left / X converted
                        self.file_list.SetItem(idx, 2, f"0 / {fi.pages}")
                        self.file_list.SetItem(idx, 3, fi.status)
                        self.file_list.SetItemTextColour(idx, self.C_DONE)
                    else:
                        fi.status = "Ready"
                        # X left / 0 converted
                        self.file_list.SetItem(idx, 2, f"{fi.pages} / 0")
                        self.file_list.SetItem(idx, 3, fi.status)
                        self.file_list.SetItemTextColour(idx, self.C_READY)
                else:
                    fi.status = "Error"
                    self.file_list.SetItem(idx, 2, "—")
                    self.file_list.SetItem(idx, 3, fi.error or "Error")
                    self.file_list.SetItemTextColour(idx, self.C_ERROR)
                break

        # recalculate running sums and update status labels
        self._total_files_all = len(self.file_info)
        self._total_pages_all = sum(fi.pages for fi in self.file_info.values())
        self.lbl_pdf_left.SetLabel(f"PDFs left: {self._total_files_all}")
        self.lbl_pages_left.SetLabel(f"Pages left: {self._total_pages_all}")

    # ==================================================================== #
    #  Conversion
    # ==================================================================== #
    def _on_start(self, _evt: wx.CommandEvent) -> None:
        # Post-process-only mode branches off before any PDF work.
        if self.chk_post_only.GetValue():
            self._start_post_process()
            return

        # gather files eligible for conversion (both newly added and previously converted)
        pdf_files: List[str] = []
        for idx, fi in self.file_info.items():
            if fi.status in ("Ready", "Done"):
                pdf_files.append(fi.path)

        if not pdf_files:
            wx.MessageBox("No PDF files found to convert.", "Error", wx.OK | wx.ICON_ERROR)
            return  # UI stays enabled — user can add files

        save_images = self.chk_images.GetValue()
        save_drawings = self.chk_drawings.GetValue()
        save_formulas = self.chk_formulas.GetValue()

        # remove stale .txt so assemble_results rewrites from scratch
        for pdf_path in pdf_files:
            txt_path = os.path.splitext(pdf_path)[0] + ".txt"
            if os.path.isfile(txt_path):
                os.remove(txt_path)

        # build config
        config = ConversionConfig(
            save_images=save_images,
            save_drawings=save_drawings,
            save_formulas=save_formulas,
            enable_margin_check=self.chk_margin.GetValue(),
            hyphen_fix_enabled=self.chk_hyphen.GetValue(),
            include_metadata=self.chk_metadata.GetValue(),
            min_size_px=self.spin_min_size.GetValue(),
            max_items_per_page=self.spin_max_items.GetValue(),
        )

        # compute totals for progress
        self._total_pages = sum(fi.pages for fi in self.file_info.values() if fi.status in ("Ready", "Done"))
        self._total_files = len(pdf_files)
        self._pages_done = 0
        self._files_done = 0
        self._images_saved = 0
        self._drawings_saved = 0
        self._formulas_saved = 0
        self._per_file_pages_done.clear()
        self._per_file_images.clear()
        self._per_file_drawings.clear()
        self._per_file_formulas.clear()
        self._start_time = time.time()
        self._is_processing = True

        # mark all as processing
        for idx, fi in self.file_info.items():
            if fi.path in pdf_files:
                fi.status = "Processing"
                fi.pages_done = 0
                # pages left / 0 converted
                self.file_list.SetItem(idx, 2, f"{fi.pages} / 0")
                self.file_list.SetItem(idx, 3, fi.status)
                self.file_list.SetItemTextColour(idx, self.C_PROCESS)

        # disable UI
        self._enable_ui(False)

        # gui_example pattern: clear progress log before starting
        self.log_ctrl.Clear()

        self._log("Starting conversion …")
        self.progress.SetValue(0)
        self._stop_flag.clear()

        # start worker
        self.converter_thread = ConverterThread(
            pdf_files=pdf_files,
            output_folder=".",  # dummy – we use source dirs
            cores=self.spin_cores.GetValue(),
            config=config,
            progress_callback=self._throttled_progress,
            finish_callback=self._on_finish,
            stop_flag=self._stop_flag,
        )
        self.converter_thread.start()

    # ---- post-process-only mode ----
    def _start_post_process(self) -> None:
        """Run standalone post-processing over the directories of listed files.

        The source PDFs are never opened — we only use each file's folder to
        locate the already-converted ``.txt`` + ``_media`` output. Works even on
        rows whose PDF is gone, as long as the .txt is still there.
        """
        directories: List[str] = []
        seen = set()
        for fi in self.file_info.values():
            folder = os.path.dirname(fi.path)
            if folder and folder not in seen:
                seen.add(folder)
                directories.append(folder)

        if not directories:
            wx.MessageBox(
                "Add at least one file first — post-processing uses its folder "
                "to find the .txt and _media output.",
                "Nothing to post-process", wx.OK | wx.ICON_INFORMATION,
            )
            return

        self._enable_ui(False)
        self.log_ctrl.Clear()
        self._log("Starting post-processing …")
        self.progress.Pulse()  # indeterminate — no per-page count in this mode

        self.post_process_thread = PostProcessThread(
            directories=directories,
            finish_callback=self._on_post_process_finish,
            log_callback=lambda msg: wx.CallAfter(self._log, msg),
        )
        self.post_process_thread.start()

    def _on_post_process_finish(self, totals: Dict[str, int]) -> None:
        """Worker-thread callback → push summary to the GUI thread."""
        wx.CallAfter(self._finish_post_process, totals)

    def _finish_post_process(self, totals: Dict[str, int]) -> None:
        processed = totals.get("files_processed", 0)
        seen = totals.get("files_seen", 0)
        added = totals.get("first_pass_additions", 0)
        updated = totals.get("second_pass_updates", 0)
        skipped = totals.get("files_skipped_no_media", 0)
        missing = totals.get("files_not_found", 0)
        errors = totals.get("errors_occurred", 0)
        self._log(
            f"Post-processing done — {processed}/{seen} files, "
            f"{added} descriptions added, {updated} updated, "
            f"{skipped} skipped (no media), {missing} missing, {errors} errors"
        )
        self.progress.SetValue(0)
        self._enable_ui(True)
        self.post_process_thread = None

    def _on_cancel(self, _evt: wx.CommandEvent) -> None:
        # Signal cancellation and return immediately — the worker thread
        # notices the flag between chunks and fires the normal finish
        # callback, which re-enables the UI. Never join() on the GUI thread.
        if self.converter_thread:
            self._log("[INFO] Cancelling — waiting for running chunks to finish…")
            self._stop_flag.set()
            self.btn_cancel.Enable(False)

    def _throttled_progress(
        self,
        pdf_filename: str,
        pages_done: int,
        total_pages: int,
        images_saved: int,
        drawings_saved: int,
        formulas_saved: int = 0,
    ) -> None:
        """Throttled wrapper: limits progress updates to ~10/sec.

        The worker thread fires the callback once per page; without throttling
        that floods wx.CallAfter with thousands of events.  We gate on a
        100 ms minimum interval so the GUI stays responsive.
        """
        now = time.time()
        if now - self._last_progress_update < 0.1:
            return
        self._last_progress_update = now
        self._on_progress(
            pdf_filename=pdf_filename,
            pages_done=pages_done,
            total_pages=total_pages,
            images_saved=images_saved,
            drawings_saved=drawings_saved,
            formulas_saved=formulas_saved,
        )

    def _on_progress(
        self,
        pdf_filename: str,
        pages_done: int,
        total_pages: int,
        images_saved: int,
        drawings_saved: int,
        formulas_saved: int = 0,
    ) -> None:
        """Called from the worker thread after every page.

        Uses wx.CallAfter to push every update to the GUI thread so the
        UI stays responsive and no race conditions occur.
        """
        wx.CallAfter(
            self._update_gui,
            pages_done=pages_done,
            total_pages=total_pages,
            images_saved=images_saved,
            drawings_saved=drawings_saved,
            formulas_saved=formulas_saved,
            pdf_filename=pdf_filename,
        )

    def _update_gui(
        self,
        pages_done: int,
        total_pages: int,
        images_saved: int,
        drawings_saved: int,
        formulas_saved: int = 0,
        pdf_filename: str = "", 
    ) -> None:
        """GUI‑thread update – called via wx.CallAfter.

        Tracks per‑file progress and updates all counters.
        """
        # pages_done, images_saved, drawings_saved, formulas_saved are all per‑file cumulative values
        self._per_file_pages_done[pdf_filename] = pages_done
        self._per_file_images[pdf_filename] = images_saved
        self._per_file_drawings[pdf_filename] = drawings_saved
        self._per_file_formulas[pdf_filename] = formulas_saved

        # update per‑file progress display
        for idx, fi in self.file_info.items():
            if fi.name == pdf_filename:
                fi.pages_done = pages_done
                # pages left / pages converted
                self.file_list.SetItem(idx, 2, f"{fi.pages - fi.pages_done} / {fi.pages_done}")
                self.file_list.SetItem(idx, 4, str(images_saved))
                self.file_list.SetItem(idx, 5, str(drawings_saved))
                self.file_list.SetItem(idx, 6, str(formulas_saved))
                break

        # update cumulative totals shown in the status panel (sum of all files)
        self._images_saved = sum(self._per_file_images.values())
        self._drawings_saved = sum(self._per_file_drawings.values())
        self._formulas_saved = sum(self._per_file_formulas.values())

        # global pages_done = sum of all per‑file pages_done
        self._pages_done = sum(self._per_file_pages_done.values())

        # update status panel labels with cumulative totals
        self.lbl_images.SetLabel(f"Images saved: {self._images_saved}")
        self.lbl_drawings.SetLabel(f"Drawings saved: {self._drawings_saved}")
        self.lbl_formulas.SetLabel(f"Formulas saved: {self._formulas_saved}")

        # ---- rolling 1‑second pages/sec mean ----
        now = time.time()
        self._pps_window.append((now, self._pages_done))
        # prune entries older than 1 s
        cutoff = now - 1.0
        while self._pps_window and self._pps_window[0][0] < cutoff:
            self._pps_window.popleft()

        if len(self._pps_window) >= 2:
            # use the FIRST and LAST entry for a proper per‑interval rate
            t0, p0 = self._pps_window[0]
            t1, p1 = self._pps_window[-1]
            dt = t1 - t0
            pps = (p1 - p0) / dt if dt > 0 else 0.0
        else:
            pps = 0.0

        # mark files that have reached their page count as Done
        for idx, fi in self.file_info.items():
            if fi.status == "Processing" and fi.pages_done >= fi.pages:
                fi.status = "Done"
                # 0 left / pages converted
                self.file_list.SetItem(idx, 2, f"0 / {fi.pages}")
                self.file_list.SetItem(idx, 3, fi.status)
                self.file_list.SetItemTextColour(idx, self.C_DONE)

        # ---- files left: derived from actual statuses, never a counter ----
        self._files_done = sum(
            1 for fi in self.file_info.values() if fi.status == "Done"
        )
        files_left = max(0, self._total_files - self._files_done)

        # ---- ETA: estimated remaining vs real elapsed ----
        remaining_pages = self._total_pages - self._pages_done
        eta_secs = remaining_pages / pps if pps > 0 else 0.0
        elapsed = time.time() - self._start_time

        self.lbl_pdf_left.SetLabel(f"PDFs left: {files_left}")
        self.lbl_pages_left.SetLabel(f"Pages left: {remaining_pages}")
        self.lbl_pps.SetLabel(f"Pages/sec: {pps:.1f}")
        self.lbl_eta.SetLabel(f"ETA: {_fmt_time(eta_secs)} | Real: {_fmt_time(elapsed)}")
        self.lbl_images.SetLabel(f"Images saved: {self._images_saved}")
        self.lbl_drawings.SetLabel(f"Drawings saved: {self._drawings_saved}")
        self.lbl_formulas.SetLabel(f"Formulas saved: {self._formulas_saved}")

        pct = int(self._pages_done / self._total_pages * 100) if self._total_pages else 0
        self.progress.SetValue(pct)

    def _on_finish(self, event: wx.PyEvent) -> None:
        """Called from the worker thread when conversion completes.

        Pushes the actual GUI update to the GUI thread via wx.CallAfter.
        """
        wx.CallAfter(
            self._finish_gui,
            success=event.success,
            elapsed=event.elapsed,
            pages=event.pages,
            pps=event.pps,
            stats=event.stats,
        )

    def _finish_gui(
        self,
        success: bool,
        elapsed: float,
        pages: int,
        pps: float,
        stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """GUI‑thread callback applied when conversion finishes."""
        if success:
            self._log(
                f"Done — {pages} pages in {elapsed:.2f}s "
                f"({pps:.1f} p/s)"
            )
            self.progress.SetValue(self.progress.GetRange())

        per_file = (stats or {}).get("per_file", {})

        # Track if any files had low character density
        low_char_files = []

        # force every remaining Processing file to Done and write final per‑file counts
        for idx, fi in self.file_info.items():
            if fi.status == "Processing":
                fi.status = "Done"
                # 0 left / pages converted
                self.file_list.SetItem(idx, 2, f"0 / {fi.pages}")
                self.file_list.SetItem(idx, 3, fi.status)
                self.file_list.SetItemTextColour(idx, self.C_DONE)

            # write final images / drawings / formulas for this file
            if fi.name in per_file:
                fs = per_file[fi.name]
                self.file_list.SetItem(idx, 4, str(fs.get("images_saved", 0)))
                self.file_list.SetItem(idx, 5, str(fs.get("drawings_saved", 0)))
                self.file_list.SetItem(idx, 6, str(fs.get("formulas_saved", 0)))

                # --- New Logic: Check Character Density ---
                char_count = fs.get("char_count", 0)
                pages_processed = fs.get("pages_processed", 0)
                
                if pages_processed > 0:
                    avg_chars = char_count / pages_processed
                    if avg_chars < 100:
                        fi.status = "Few Chars"
                        self.file_list.SetItem(idx, 3, fi.status)
                        self.file_list.SetItemTextColour(idx, self.C_FEW_CHARS)
                        low_char_files.append((fi.name, avg_chars))
                # -------------------------------------------

        # Log messages for low character density files
        if low_char_files:
            for name, avg in low_char_files:
                self._log(f"Warning: '{name}' has low text density ({avg:.1f} chars/page).")

        # sync final counts so status labels show zero
        self._pages_done = self._total_pages
        self.lbl_pdf_left.SetLabel(f"PDFs left: 0/{self._total_files}")
        self.lbl_pages_left.SetLabel(f"Pages left: 0/{self._total_pages}")
        self.lbl_pps.SetLabel(f"Pages/sec: {pps:.1f}")
        self.lbl_eta.SetLabel(f"ETA: 0s | Real: {_fmt_time(elapsed)}")

        self._is_processing = False
        self._enable_ui(True)
        self.converter_thread = None


    def _enable_ui(self, enable: bool) -> None:
        self.btn_start.Enable(enable)
        self.btn_cancel.Enable(not enable)
        self.btn_add.Enable(enable)
        self.btn_add_folder.Enable(enable)
        self.btn_remove.Enable(enable)
        self.btn_clear.Enable(enable)
        for chk in (self.chk_images, self.chk_drawings, self.chk_formulas, self.chk_margin,
                     self.chk_hyphen, self.chk_metadata, self.chk_post_only):
            chk.Enable(enable)
        self.spin_cores.Enable(enable)
        self.spin_min_size.Enable(enable)
        self.spin_max_items.Enable(enable)

    # ---- helpers ----
    def _log(self, msg: str) -> None:
        self.log_ctrl.AppendText(msg + "\n")
        print(msg, flush=True)

    def _show_help(self) -> None:
        """Display the help/info text in the converted-text viewer."""
        self.txt_viewer.SetValue(HELP_TEXT)

# --------------------------------------------------------------------------- #
#  Time formatter
# --------------------------------------------------------------------------- #
def _fmt_time(secs: float) -> str:
    """Format seconds as HH:MM:SS or a short form."""
    if secs < 60:
        return f"{secs:.0f}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{m:.0f}m {s:.0f}s"
    h, m = divmod(m, 60)
    return f"{h:.0f}h {m:.0f}m"

# --------------------------------------------------------------------------- #
#  Application entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    app = wx.App()
    frame = PDFParserFrame()
    frame.Show()
    app.MainLoop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()





