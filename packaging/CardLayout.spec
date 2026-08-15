from pathlib import Path

project_root = Path(SPECPATH).parent

# Standard PyInstaller hooks and static import analysis provide the PySide6,
# OpenCV, NumPy, Pillow, and PyMuPDF binaries/plugins used by CardLayout. Do not
# collect all PyMuPDF submodules: its unrelated command-line tools pull in a
# large scientific/notebook stack that the desktop application never uses.

a = Analysis(
    [str(project_root / "cardlayout" / "__main__.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "tests",
        "testing",
        # Optional PyMuPDF table/notebook integrations are not used by
        # CardLayout and otherwise pull hundreds of megabytes into the build.
        "pandas",
        "scipy",
        "matplotlib",
        "IPython",
        "jupyter",
        "jupyter_client",
        "nbformat",
        "notebook",
        "traitlets",
        "zmq",
        "psutil",
    ],
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
    name="CardLayout",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / "icon" / "credit-card.ico"),
)
