from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from cardlayout.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("CardLayout")
    app.setOrganizationName("CardLayout")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
