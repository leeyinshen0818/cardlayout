from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QFrame, QLabel, QMessageBox
from PIL import Image

from cardlayout.models.card_side import CardSide
from cardlayout.models.detection import CardDetectionResult
from cardlayout.models.perspective import PerspectiveResult
import cardlayout.ui.main_window as main_window_module
import cardlayout.__main__ as app_module
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


def test_application_starts_maximized(monkeypatch) -> None:
    events: list[str] = []

    class _Application:
        def __init__(self, arguments) -> None:
            del arguments

        def setApplicationName(self, name: str) -> None:
            del name

        def setOrganizationName(self, name: str) -> None:
            del name

        def setWindowIcon(self, icon) -> None:
            assert icon == "application-icon"
            events.append("application-icon")

        def exec(self) -> int:
            return 0

    class _Window:
        def setWindowIcon(self, icon) -> None:
            assert icon == "application-icon"
            events.append("window-icon")

        def showMaximized(self) -> None:
            events.append("maximized")

    monkeypatch.setattr(app_module, "QApplication", _Application)
    monkeypatch.setattr(app_module, "MainWindow", _Window)
    monkeypatch.setattr(app_module, "QIcon", lambda path: "application-icon")

    assert app_module.main() == 0
    assert events == ["application-icon", "window-icon", "maximized"]


def test_compact_ui_density_tracks_logical_screen_resolution() -> None:
    assert MainWindow._uses_compact_ui(1600, 900)
    assert MainWindow._uses_compact_ui(1536, 864)
    assert MainWindow._uses_compact_ui(1280, 720)
    assert not MainWindow._uses_compact_ui(1920, 1080)


def test_compact_corrections_sidebar_uses_smaller_tiles() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    if window.compact_ui:
        assert window.corrections_sidebar.maximumWidth() == 280
        for button in window.corrections_sidebar._buttons.values():
            assert button.width() == 104
            assert button.height() == 66
            assert button.iconSize().width() == 68
            assert button.iconSize().height() == 40
    window.close()


def test_main_window_uses_clean_responsibility_based_layout() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()

    header = window.findChild(QFrame, "header")
    assert header is not None
    assert [label.text() for label in header.findChildren(QLabel)] == ["CardLayout"]
    assert not any(
        label.text() == "Click Front or Back to select and edit it"
        for label in window.findChildren(QLabel)
    )
    assert window.pdf_button.parent() is not header
    assert window.swap_button.parent() is not header
    assert window.pdf_button.geometry().top() < window.front_widget.geometry().top()
    assert window.front_widget.geometry().bottom() < window.swap_button.geometry().top()
    assert window.swap_button.geometry().bottom() < window.back_widget.geometry().top()
    assert window.swap_button.text() == "⇅  Swap Front / Back"

    window.close()


def test_exports_save_directly_to_downloads_without_save_dialog(
    tmp_path: Path, monkeypatch
) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    saved_paths: list[Path] = []
    opened_folders: list[Path] = []

    monkeypatch.setattr(window, "_downloads_directory", lambda: tmp_path)
    monkeypatch.setattr(
        window.jpg_exporter,
        "export",
        lambda path, front, back: saved_paths.append(Path(path)),
    )
    monkeypatch.setattr(
        window.pdf_exporter,
        "export",
        lambda path, front, back: saved_paths.append(Path(path)),
    )
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: pytest.fail("Save As dialog must not open"),
    )
    monkeypatch.setattr(
        main_window_module.QDesktopServices,
        "openUrl",
        lambda url: opened_folders.append(Path(url.toLocalFile())) or True,
    )

    (tmp_path / "card-layout.jpg").touch()
    window._export_jpg()
    window._export_pdf()

    assert saved_paths == [
        tmp_path / "card-layout (1).jpg",
        tmp_path / "card-layout.pdf",
    ]
    assert opened_folders == [tmp_path.resolve(), tmp_path.resolve()]
    assert "exported to" in window.statusBar().currentMessage()
    window.close()


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


def test_side_panel_is_clean_and_corrections_reset_without_removing_file(
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
    assert window.front.rectified_image is not None
    original_bytes = window.front.original_image.tobytes()
    automatic = window.front.automatic_perspective_result
    assert automatic is not None and automatic.rectified_image is not None
    automatic_bytes = automatic.rectified_image.tobytes()

    assert not hasattr(window.front_widget, "original_button")
    assert not hasattr(window.front_widget, "detected_button")
    assert not hasattr(window.front_widget, "corrected_button")
    assert not hasattr(window.front_widget, "redetect_button")
    assert not hasattr(window.front_widget, "reset_detection_button")
    assert not hasattr(window.front_widget, "reset_correction_button")
    assert window.front_widget.choose_button.text() == "Choose Front"
    assert window.front_widget.clear_button.text() == "Clear"
    assert window.front_widget.adjust_corners_button.text() == "Adjust Corners"
    assert window.front_widget.reset_button.text() == "Reset"

    assert not window.corrections_sidebar.isVisible()
    preview_width_closed = window.preview_panel.width()
    _click_card(window, "front")
    app.processEvents()
    assert window.corrections_sidebar.isVisible()
    assert window.corrections_sidebar.title.text() == "Corrections · FRONT"
    popover = window.corrections_sidebar
    assert popover.parent() is window.splitter
    assert window.preview_panel.width() < preview_width_closed
    assert popover.geometry().left() >= window.preview_panel.geometry().right()
    assert window.preview._page_rect().center().x() == pytest.approx(
        window.preview.width() / 2, abs=1
    )
    assert set(popover._buttons) == {
        ("sharpen", "soft"),
        ("sharpen", "normal"),
        ("sharpen", "sharp"),
        ("sharpen", "sharper"),
        ("tone", "normal"),
        ("tone", "bright_10"),
        ("tone", "bright_20"),
        ("tone", "bright_contrast"),
        ("tone", "strong_bright_contrast"),
    }

    QTest.mouseClick(
        popover._buttons[("sharpen", "sharp")], Qt.MouseButton.LeftButton
    )
    app.processEvents()
    assert window.front.image_correction_state.sharpen == "sharp"
    assert window.front.best_image.tobytes() != automatic_bytes
    assert window.front.original_image.tobytes() == original_bytes
    assert window.front_widget.reset_button.isEnabled()

    QTest.mouseClick(window.front_widget.reset_button, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert window.front is not None
    assert window.front.image_correction_state.is_normal
    assert window.front.detection_result is not None
    assert window.front.detected_image is not None
    assert window.front.best_image.tobytes() == automatic_bytes

    QTest.mouseClick(window.front_widget.clear_button, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert window.front is None
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


def test_front_and_back_correction_tiles_update_independently(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    front_image = Image.new("RGB", (428, 270), (110, 135, 160))
    back_image = Image.new("RGB", (428, 270), (145, 115, 90))
    front = CardSide(
        side="front",
        source_path=tmp_path / "front.png",
        source_type="image",
        source_page=None,
        original_image=front_image,
        processed_image=front_image.copy(),
    )
    back = CardSide(
        side="back",
        source_path=tmp_path / "back.png",
        source_type="image",
        source_page=None,
        original_image=back_image,
        processed_image=back_image.copy(),
    )
    window = MainWindow()
    window.front, window.back = front, back
    window._refresh()
    window.show()
    app.processEvents()

    QTest.mouseClick(window.front_widget.preview, Qt.MouseButton.LeftButton)
    QTest.mouseClick(
        window.corrections_sidebar._buttons[("sharpen", "sharp")],
        Qt.MouseButton.LeftButton,
    )
    app.processEvents()
    assert front.image_correction_state.sharpen == "sharp"
    assert back.image_correction_state.is_normal

    QTest.mouseClick(
        window.corrections_sidebar.close_button, Qt.MouseButton.LeftButton
    )
    app.processEvents()
    assert not window.corrections_sidebar.isVisible()
    assert front.image_correction_state.sharpen == "sharp"
    _click_card(window, "back")
    assert window.corrections_sidebar.title.text() == "Corrections · BACK"
    QTest.mouseClick(
        window.corrections_sidebar._buttons[
            ("tone", "bright_contrast")
        ],
        Qt.MouseButton.LeftButton,
    )
    app.processEvents()
    assert front.image_correction_state.sharpen == "sharp"
    assert front.image_correction_state.tone == "normal"
    assert back.image_correction_state.sharpen == "normal"
    assert back.image_correction_state.tone == "bright_contrast"
    window.close()


def test_sidebar_reflows_preview_switches_side_and_preserves_state(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    front_image = Image.new("RGB", (428, 270), (105, 135, 165))
    back_image = Image.new("RGB", (428, 270), (70, 115, 160))
    window = MainWindow()
    window.front = CardSide(
        side="front",
        source_path=tmp_path / "front.png",
        source_type="image",
        source_page=None,
        original_image=front_image,
        processed_image=front_image.copy(),
    )
    window.back = CardSide(
        side="back",
        source_path=tmp_path / "back.png",
        source_type="image",
        source_page=None,
        original_image=back_image,
        processed_image=back_image.copy(),
    )
    window._refresh()
    window.show()
    app.processEvents()

    layout_before = window.engine.calculate()
    closed_width = window.preview_panel.width()
    assert not window.corrections_sidebar.isVisible()
    assert window.corrections_button.isVisible()

    QTest.mouseClick(window.corrections_button, Qt.MouseButton.LeftButton)
    app.processEvents()
    expanded_width = window.preview_panel.width()
    assert window.preview.selected_side == "front"
    assert window.corrections_sidebar.title.text() == "Corrections · FRONT"
    assert window.corrections_sidebar.isVisible()
    assert not window.corrections_button.isVisible()
    assert expanded_width < closed_width
    assert window.corrections_sidebar.geometry().left() >= (
        window.preview_panel.geometry().right()
    )

    _click_card(window, "back")
    assert window.preview.selected_side == "back"
    assert window.corrections_sidebar.isVisible()
    assert window.corrections_sidebar.title.text() == "Corrections · BACK"
    QTest.mouseClick(
        window.corrections_sidebar._buttons[("tone", "bright_10")],
        Qt.MouseButton.LeftButton,
    )
    app.processEvents()
    assert window.back.image_correction_state.tone == "bright_10"
    assert window.front.image_correction_state.is_normal

    layout_before_reset = window.engine.calculate()
    QTest.mouseClick(
        window.corrections_sidebar.reset_button, Qt.MouseButton.LeftButton
    )
    app.processEvents()
    assert window.back.image_correction_state.is_normal
    assert window.corrections_sidebar.isVisible()
    assert window.preview.selected_side == "back"
    assert not window.corrections_sidebar.reset_button.isEnabled()
    assert window.engine.calculate() == layout_before_reset

    QTest.mouseClick(
        window.corrections_sidebar._buttons[("tone", "bright_10")],
        Qt.MouseButton.LeftButton,
    )
    app.processEvents()
    assert window.back.image_correction_state.tone == "bright_10"

    QTest.mouseClick(
        window.corrections_sidebar.close_button, Qt.MouseButton.LeftButton
    )
    app.processEvents()
    assert not window.corrections_sidebar.isVisible()
    assert window.corrections_button.isVisible()
    assert window.preview.selected_side is None
    assert window.preview_panel.width() > expanded_width
    assert window.back.image_correction_state.tone == "bright_10"
    layout_after = window.engine.calculate()
    assert layout_after.front == layout_before.front
    assert layout_after.back == layout_before.back
    assert window.preview._page_rect().center().x() == pytest.approx(
        window.preview.width() / 2, abs=1
    )

    QTest.mouseClick(window.corrections_button, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert window.corrections_sidebar.isVisible()
    assert window.preview.selected_side == "front"

    window.resize(1050, 720)
    app.processEvents()
    assert window.corrections_sidebar.geometry().left() >= (
        window.preview_panel.geometry().right()
    )
    assert window.preview.width() >= window.preview.minimumWidth()
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
