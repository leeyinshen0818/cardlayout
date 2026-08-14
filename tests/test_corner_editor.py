from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from cardlayout.models.card_size import MALAYSIA_IC
from cardlayout.services.perspective_corrector import PerspectiveCorrector
from cardlayout.ui.corner_editor import CornerEditorDialog


AUTOMATIC = ((80.0, 60.0), (720.0, 80.0), (700.0, 460.0), (100.0, 440.0))


def _dialog() -> CornerEditorDialog:
    return CornerEditorDialog(
        Image.new("RGB", (800, 520), (80, 100, 130)),
        AUTOMATIC,
        PerspectiveCorrector(MALAYSIA_IC),
        0.85,
    )


def test_all_four_corner_handles_move_independently_in_image_coordinates() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = _dialog()
    dialog.show()
    app.processEvents()

    expected = []
    for index, handle in enumerate(dialog.canvas._handles):
        point = QPointF(120 + index * 130, 95 + index * 75)
        handle.setPos(point)
        expected.append((point.x(), point.y()))
    app.processEvents()

    assert dialog.canvas.corners == tuple(expected)
    assert all(
        handle.flags() & handle.GraphicsItemFlag.ItemIsMovable
        for handle in dialog.canvas._handles
    )
    dialog.close()


def test_zoom_fit_and_pan_transform_never_change_corner_coordinates() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = _dialog()
    dialog.show()
    app.processEvents()
    original = dialog.canvas.corners

    QTest.mouseClick(dialog.actual_button, Qt.MouseButton.LeftButton)
    assert dialog.canvas.transform().m11() == 1.0
    dialog.canvas.scale(2.0, 2.0)
    dialog.canvas.horizontalScrollBar().setValue(70)
    dialog.canvas.verticalScrollBar().setValue(40)
    assert dialog.canvas.corners == original

    QTest.mouseClick(dialog.fit_button, Qt.MouseButton.LeftButton)
    assert dialog.canvas.corners == original
    dialog.close()


def test_invalid_polygon_disables_apply_and_reset_restores_automatic_points() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = _dialog()
    dialog.show()
    app.processEvents()

    crossing = ((80.0, 60.0), (700.0, 460.0), (720.0, 80.0), (100.0, 440.0))
    dialog.canvas.set_corners(crossing)
    app.processEvents()
    assert not dialog.apply_button.isEnabled()
    assert "cross" in dialog.warning.text().lower()

    QTest.mouseClick(dialog.reset_button, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert dialog.canvas.corners == AUTOMATIC
    assert dialog.apply_button.isEnabled()
    dialog.close()


def test_corner_handles_are_clamped_to_source_image_boundaries() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = _dialog()
    dialog.show()
    app.processEvents()
    dialog.canvas._handles[0].setPos(-500, -300)
    dialog.canvas._handles[2].setPos(1200, 900)
    app.processEvents()
    assert dialog.canvas.corners[0] == (0.0, 0.0)
    assert dialog.canvas.corners[2] == (799.0, 519.0)
    dialog.close()


def test_apply_produces_a_manual_rectification_result() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = _dialog()
    dialog.show()
    app.processEvents()

    QTest.qWait(120)
    assert not dialog.preview.pixmap().isNull()

    QTest.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert dialog.result is not None
    assert dialog.result.success
    assert dialog.result.method == "manual"
    assert dialog.result.rectified_image is not None
