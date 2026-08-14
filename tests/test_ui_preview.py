from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox
from PIL import Image

from cardlayout.models.card_side import CardSide
from cardlayout.models.detection import CardDetectionResult
from cardlayout.models.perspective import PerspectiveResult
import cardlayout.ui.main_window as main_window_module
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
    assert window.front.rectified_image is not None
    assert window.front_widget._preview_mode == "corrected"

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


def test_redetect_requires_confirmation_before_replacing_manual_corners(
    tmp_path: Path, monkeypatch
) -> None:
    app = QApplication.instance() or QApplication([])
    pixels = np.full((600, 800, 3), (45, 55, 65), dtype=np.uint8)
    box = cv2.boxPoints(((400, 300), (320, 202), 12)).astype(np.int32)
    cv2.fillConvexPoly(pixels, box, (225, 235, 245))
    cv2.polylines(pixels, [box], True, (10, 10, 10), 5)
    path = tmp_path / "manual-redetect.png"
    Image.fromarray(pixels).save(path)
    window = MainWindow()
    window._load_side(str(path), "front")
    assert window.front is not None
    assert window.front.detection_result is not None
    manual = window.card_processor.apply_manual_correction(
        window.front, window.front.detection_result.polygon_points
    )
    assert manual.success and window.front.has_manual_correction
    window.engine.adjust_vertical_offset("front", 5.0)

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.No,
    )
    window._redetect_side("front")
    assert window.front.has_manual_correction

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )
    window._redetect_side("front")
    assert not window.front.has_manual_correction
    assert window.front.automatic_perspective_result is not None
    assert window.engine.vertical_offset("front") == 5.0
    window.close()


def test_corner_editor_starts_from_refined_automatic_corners(
    monkeypatch, tmp_path: Path
) -> None:
    app = QApplication.instance() or QApplication([])
    image = Image.new("RGB", (800, 600), (80, 100, 120))
    rough = ((100.0, 100.0), (700.0, 110.0), (690.0, 490.0), (105.0, 480.0))
    refined = ((106.0, 104.0), (694.0, 114.0), (685.0, 484.0), (110.0, 476.0))
    side = CardSide(
        side="front",
        source_path=tmp_path / "refined-start.png",
        source_type="image",
        source_page=None,
        original_image=image,
        processed_image=image.copy(),
    )
    side.apply_detection(
        CardDetectionResult(
            success=True,
            confidence=0.9,
            confidence_level="high",
            polygon_points=rough,
            cropped_image=image.crop((100, 100, 700, 490)),
        )
    )
    side.apply_automatic_correction(
        PerspectiveResult(
            success=True,
            source_points=rough,
            refined_points=refined,
            rectified_image=Image.new("RGB", (856, 540), "white"),
            confidence=0.9,
            confidence_level="high",
            status="corrected",
            method="automatic",
        )
    )
    captured = {}

    class _CancelledEditor:
        result = None

        def __init__(self, image, automatic_points, corrector, confidence, current_points, parent):
            del image, corrector, confidence, parent
            captured["automatic"] = automatic_points
            captured["current"] = current_points

        def exec(self):
            return 0

    monkeypatch.setattr(main_window_module, "CornerEditorDialog", _CancelledEditor)
    window = MainWindow()
    window.front = side
    window._adjust_corners("front")

    assert captured["automatic"] == refined
    assert captured["current"] == refined
    window.close()
    app.processEvents()
