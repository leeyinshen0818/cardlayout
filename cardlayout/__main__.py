from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from cardlayout.ui.main_window import MainWindow


def _resource_path(*parts: str) -> Path:
    """Resolve packaged resources both from source and PyInstaller one-file."""
    frozen_root = getattr(sys, "_MEIPASS", None)
    root = Path(frozen_root) if frozen_root else Path(__file__).resolve().parents[1]
    return root.joinpath(*parts)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("CardLayout")
    app.setOrganizationName("CardLayout")
    icon = QIcon(str(_resource_path("icon", "credit-card.ico")))
    app.setWindowIcon(icon)
    window = MainWindow()
    window.setWindowIcon(icon)
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
