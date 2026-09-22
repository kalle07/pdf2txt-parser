"""Build the PDF Parser desktop application with PyInstaller.

Run this script from the project directory with:

    python build.py

The finished Windows application is written to ``dist/PDFParser.exe``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ENTRY_POINT = PROJECT_ROOT / "main.py"
APP_NAME = "PDFParser-by-kalle07"


def main() -> None:
    """Package the wxPython application as a single Windows executable."""
    if not ENTRY_POINT.is_file():
        raise FileNotFoundError(f"Application entry point not found: {ENTRY_POINT}")

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        # '--console',
        "--name",
        APP_NAME,
        "--distpath",
        str(PROJECT_ROOT / "dist"),
        "--workpath",
        str(PROJECT_ROOT / "build"),
        "--specpath",
        str(PROJECT_ROOT / "build"),
        "--paths",
        str(PROJECT_ROOT),
        "--hidden-import",
        "wx.lib.newevent",
        # The application uses Table.extract(), not PyMuPDF's optional
        # Table.to_pandas() helper. Excluding pandas keeps PyInstaller from
        # recursively bundling the unrelated scientific/ML stack.
        "--exclude-module",
        "pandas",
        str(ENTRY_POINT),
    ]

    print("Building PDFParser.exe...")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    print(f"Build complete: {PROJECT_ROOT / 'dist' / f'{APP_NAME}.exe'}")


if __name__ == "__main__":
    main()