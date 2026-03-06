import sys
import subprocess

# Hier wird der Pfad zum Entry-Point (dein Skript) definiert
entry_point = "PDF_Parser-Sevenof9_v7i.py"  # Passe dies nach Bedarf an


# Der Befehl, der an PyInstaller übergeben wird
cmd = [
    "pyinstaller",  # Use pyinstaller directly instead of sys.executable -m PyInstaller
    "--onefile",
    "--noconfirm",
    "--clean",
    "--noconsole",  # Keine Konsole anzeigen (wichtig für GUI-Programme)
    
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
    print("Kompilierung abgeschlossen.")
except subprocess.CalledProcessError as e:
    print(f"Fehler bei der Kompilierung: {e}")
