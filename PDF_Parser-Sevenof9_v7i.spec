# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['PDF_Parser-Sevenof9_v7i.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['pdfminer.six', 'joblib', 'joblib.externals.loky.backend.resource_tracker', 'pdfplumber.utils.exceptions', 'pdfminer.layout', 'pdfminer.pdfpage', 'pdfminer.pdfinterp', 'pdfminer.pdfdocument', 'pdfminer.pdfparser', 'psutil', 'multiprocessing', 'numpy', 'concurrent.futures', 'wx', 'wx.lib.pubsub', 'wx.lib.pubsub.core'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PDF_Parser-Sevenof9_v7i',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
