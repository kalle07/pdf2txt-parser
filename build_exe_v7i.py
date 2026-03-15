import sys
import subprocess

# Define the path to the entry point (your script) here
entry_point = "PDF_Parser-Sevenof9_v7i.py"  # Adjust this as needed


# Der Befehl, der an PyInstaller übergeben wird
cmd = [
    "pyinstaller",  # Use pyinstaller directly instead of sys.executable -m PyInstaller
    "--onefile",
    "--noconfirm",
    "--clean",
    "--noconsole",  # Do not show console (important for GUI programs)
    
    # External dependencies that need explicit hidden imports
    "--hidden-import", "pdfminer.six",
    "--hidden-import", "joblib",
    "--hidden-import", "joblib.externals.loky.backend.resource_tracker",
    "--hidden-import", "pdfplumber.utils.exceptions",
    "--hidden-import", "pdfminer.layout",
    "--hidden-import", "pdfminer.pdfpage",
    "--hidden-import", "pdfminer.pdfinterp",
    "--hidden-import", "pdfminer.pdfdocument",
    "--hidden-import", "pdfminer.pdfparser",
    "--hidden-import", "psutil",
    "--hidden-import", "multiprocessing",
    "--hidden-import", "numpy",
    "--hidden-import", "concurrent.futures",
    "--hidden-import", "wx",  # This is the correct import for wxPython
    "--hidden-import", "wx.lib.pubsub",
    "--hidden-import", "wx.lib.pubsub.core",
    
    entry_point
]

try:
    subprocess.run(cmd, check=True)
    print("Compilation complete.")
except subprocess.CalledProcessError as e:
    print(f"Compilation error: {e}")
