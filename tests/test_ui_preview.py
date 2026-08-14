from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from PIL import Image

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


def test_import_detects_and_original_toggle_does_not_change_layout_image(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    pixels = np.full((600, 800, 3), (45, 55, 65), dtype=np.uint8)
    box = cv2.boxPoints(((400, 300), (320, 202), 22)).astype(np.int32)
    cv2.fillConvexPoly(pixels, box, (225, 235, 245))
    cv2.polylines(pixels, [box], True, (10, 10, 10), 5)
    path = tmp_path / "phone-photo.png"
    Image.fromarray(pixels).save(path)

    window = MainWindow()
    window.show()
    app.processEvents()
    window._load_side(str(path), "front")
    app.processEvents()

    assert window.front is not None
    assert window.front.detection_result is not None
    assert window.front.detection_result.success
    assert window.front.detected_image is not None
    processed = window.front.processed_image
    assert window.front_widget._preview_mode == "detected"

    QTest.mouseClick(window.front_widget.original_button, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert window.front_widget._preview_mode == "original"
    assert window.front.processed_image is processed

    QTest.mouseClick(
        window.front_widget.reset_detection_button, Qt.MouseButton.LeftButton
    )
    app.processEvents()
    assert window.front.detected_image is None
    assert window.front.detection_result is None
    assert window.front.processed_image.size == (800, 600)
    window.close()
