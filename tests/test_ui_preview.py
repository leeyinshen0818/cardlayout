from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from cardlayout.ui.main_window import MainWindow


def _click_card(window: MainWindow, side: str) -> None:
    layout = window.engine.calculate()
    rect = layout.front if side == "front" else layout.back
    target = window.preview._card_target(rect, window.preview._page_rect())
    QTest.mouseClick(
        window.preview,
        Qt.MouseButton.LeftButton,
        pos=target.center().toPoint(),
    )
    QApplication.processEvents()


def test_preview_selection_reveals_controls_and_moves_each_side() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()

    _click_card(window, "front")
    assert window.preview.selected_side == "front"
    assert window.preview.position_controls.isVisible()
    assert window.preview.selection_label.text() == "FRONT  ·  Default position"

    QTest.mouseClick(window.preview.move_down_button, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert window.engine.vertical_offset("front") == 1.0
    assert window.preview.selection_label.text() == "FRONT  ·  1 mm down"

    _click_card(window, "back")
    QTest.mouseClick(window.preview.move_up_button, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert window.engine.vertical_offset("front") == 1.0
    assert window.engine.vertical_offset("back") == -1.0
    assert window.preview.selection_label.text() == "BACK  ·  1 mm up"

    window.preview.clear_selection()
    assert not window.preview.position_controls.isVisible()
    window.close()
