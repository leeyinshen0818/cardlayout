from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QStandardPaths, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from cardlayout.models.card_side import CardSide, SideName
from cardlayout.models.card_size import MALAYSIA_IC
from cardlayout.models.detection import CardDetectionResult
from cardlayout.models.image_correction import ImageCorrectionState
from cardlayout.models.layout import A4_LAYOUT
from cardlayout.services.card_detector import CardDetector
from cardlayout.services.card_processing import CardProcessingService
from cardlayout.services.image_exporter import ExportError, ImageExporter
from cardlayout.services.input_loader import InputLoadError, InputLoader, SUPPORTED_SUFFIXES
from cardlayout.services.layout_engine import LayoutEngine
from cardlayout.services.page_renderer import PageRenderer
from cardlayout.services.pdf_exporter import PDFExporter
from cardlayout.ui.card_input_widget import CardInputWidget
from cardlayout.ui.corrections_popover import CorrectionsPopover
from cardlayout.ui.corner_editor import CornerEditorDialog
from cardlayout.ui.page_preview import PagePreview

FILE_FILTER = "Card sources (*.jpg *.jpeg *.png *.pdf)"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.front: CardSide | None = None
        self.back: CardSide | None = None
        self.input_loader = InputLoader()
        self.card_processor = CardProcessingService(CardDetector(MALAYSIA_IC))
        self.engine = LayoutEngine(A4_LAYOUT, MALAYSIA_IC)
        self.renderer = PageRenderer(self.engine)
        self.jpg_exporter = ImageExporter(self.renderer)
        self.pdf_exporter = PDFExporter(self.renderer)

        self.setWindowTitle("CardLayout")
        self.setMinimumSize(1050, 720)
        self.resize(1240, 820)
        self.setAcceptDrops(True)
        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        title = QLabel("CardLayout")
        title.setObjectName("appTitle")

        self.pdf_button = QPushButton("Open 2-Page PDF")
        self.pdf_button.setObjectName("secondaryButton")
        self.pdf_button.clicked.connect(self._choose_two_page_pdf)
        self.swap_button = QPushButton("⇅  Swap Front / Back")
        self.swap_button.setObjectName("swapButton")
        self.swap_button.clicked.connect(self._swap_sides)

        header = QHBoxLayout()
        header.setContentsMargins(22, 10, 22, 10)
        header.addWidget(title)
        header.addStretch()

        header_widget = QFrame()
        header_widget.setObjectName("header")
        header_widget.setLayout(header)

        self.front_widget = CardInputWidget("front")
        self.back_widget = CardInputWidget("back")
        for widget in (self.front_widget, self.back_widget):
            widget.choose_requested.connect(self._choose_side)
            widget.clear_requested.connect(self._clear_side)
            widget.adjust_corners_requested.connect(self._adjust_corners)
            widget.reset_requested.connect(self._reset_side)
            widget.corrections_requested.connect(self._show_corrections)

        size_caption = QLabel("CARD SIZE")
        size_caption.setObjectName("sectionCaption")
        size_value = QLabel(MALAYSIA_IC.label)
        size_value.setObjectName("presetValue")
        size_value.setWordWrap(True)
        local_note = QLabel("Detection and export run locally. No uploads or OCR.")
        local_note.setWordWrap(True)
        local_note.setObjectName("privacyNote")
        import_caption = QLabel("IMPORT")
        import_caption.setObjectName("sectionCaption")

        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(18, 18, 18, 18)
        controls_layout.setSpacing(13)
        controls_layout.addWidget(import_caption)
        controls_layout.addWidget(self.pdf_button)
        controls_layout.addWidget(self.front_widget)
        controls_layout.addWidget(self.swap_button)
        controls_layout.addWidget(self.back_widget)
        controls_layout.addWidget(size_caption)
        controls_layout.addWidget(size_value)
        controls_layout.addWidget(local_note)
        controls_layout.addStretch()

        controls = QWidget()
        controls.setLayout(controls_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(420)

        preview_title = QLabel("A4 PORTRAIT PREVIEW")
        preview_title.setObjectName("sectionCaption")
        self.corrections_button = QPushButton("Corrections")
        self.corrections_button.setObjectName("compactExpandButton")
        self.corrections_button.setEnabled(False)
        self.corrections_button.clicked.connect(self._expand_corrections)
        self.preview = PagePreview(self.engine)
        self.preview.position_adjust_requested.connect(self._adjust_position)
        self.preview.position_reset_requested.connect(self._reset_position)
        self.preview.side_selected.connect(self._open_corrections_for_selection)
        self.preview.selection_cleared.connect(self._collapse_corrections)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        preview_heading = QHBoxLayout()
        preview_heading.addWidget(preview_title)
        preview_heading.addStretch()
        preview_heading.addWidget(self.corrections_button)
        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(18, 18, 18, 18)
        preview_layout.addLayout(preview_heading)
        preview_layout.addWidget(self.preview, 1)
        self.preview_panel = QWidget()
        self.preview_panel.setObjectName("previewPanel")
        self.preview_panel.setLayout(preview_layout)

        self.corrections_sidebar = CorrectionsPopover(
            popup=False,
            columns=2,
            show_close=True,
        )
        self.corrections_sidebar.setMinimumWidth(280)
        self.corrections_sidebar.setMaximumWidth(320)
        self.corrections_sidebar.resize(300, self.corrections_sidebar.height())
        self.corrections_sidebar.preset_selected.connect(
            self._select_correction_preset
        )
        self.corrections_sidebar.collapse_requested.connect(
            self._collapse_corrections
        )
        self.corrections_sidebar.reset_requested.connect(
            self._reset_image_corrections
        )
        self.corrections_sidebar.hide()

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(scroll)
        self.splitter.addWidget(self.preview_panel)
        self.splitter.addWidget(self.corrections_sidebar)
        self.splitter.setSizes([390, 850, 0])
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setCollapsible(2, True)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)

        export_pdf = QPushButton("Export PDF")
        export_pdf.setObjectName("primaryButton")
        export_pdf.clicked.connect(self._export_pdf)
        export_jpg = QPushButton("Export JPG · 300 DPI")
        export_jpg.setObjectName("primaryButton")
        export_jpg.clicked.connect(self._export_jpg)
        export_bar = QHBoxLayout()
        export_bar.setContentsMargins(22, 12, 22, 12)
        export_bar.addWidget(QLabel("Print exported PDF at Actual Size / 100%"))
        export_bar.addStretch()
        export_bar.addWidget(export_jpg)
        export_bar.addWidget(export_pdf)
        export_widget = QFrame()
        export_widget.setObjectName("exportBar")
        export_widget.setLayout(export_bar)

        central_layout = QVBoxLayout()
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(header_widget)
        central_layout.addWidget(self.splitter, 1)
        central_layout.addWidget(export_widget)
        central = QWidget()
        central.setLayout(central_layout)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Drop one or two JPG, PNG, or PDF files to begin")

    def _choose_side(self, side: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, f"Choose {side.title()}", "", FILE_FILTER)
        if path:
            self._load_side(path, side)  # type: ignore[arg-type]

    def _load_side(self, path: str, side: SideName) -> None:
        try:
            result = self.input_loader.load_side(path, side)
            if side == "front":
                self.front = result.card_side
            else:
                self.back = result.card_side
            detection = self._detect_cards([result.card_side])[0]
            self._refresh()
            self._card_widget(side).show_corrected()
            self.statusBar().showMessage(
                f"Loaded {result.card_side.display_name}. {self._detection_message(detection)}",
                7000,
            )
            if result.notice:
                QMessageBox.information(self, "PDF pages", result.notice)
        except InputLoadError as exc:
            self._show_error("Could not open file", str(exc))

    def _choose_two_page_pdf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF files (*.pdf)")
        if path:
            self._load_two_page_pdf(path)

    def _load_two_page_pdf(self, path: str) -> None:
        try:
            result = self.input_loader.load_two_page_pdf(path)
            self.front, self.back = result.front, result.back
            cards = [card for card in (self.front, self.back) if card is not None]
            detections = self._detect_cards(cards)
            self._refresh()
            if self.front is not None:
                self.front_widget.show_corrected()
            if self.back is not None:
                self.back_widget.show_corrected()
            detection_summary = "; ".join(
                self._detection_message(detection) for detection in detections
            )
            self.statusBar().showMessage(
                f"Loaded {Path(path).name}. {detection_summary}", 8000
            )
            if result.notice:
                QMessageBox.information(self, "PDF pages", result.notice)
        except InputLoadError as exc:
            self._show_error("Could not open PDF", str(exc))

    def _clear_side(self, side: str) -> None:
        if side == "front":
            self.front = None
        else:
            self.back = None
        self._refresh()
        self.statusBar().showMessage(f"{side.title()} cleared", 3000)

    def _swap_sides(self) -> None:
        old_front, old_back = self.front, self.back
        self.front = old_back.assigned_to("front") if old_back else None
        self.back = old_front.assigned_to("back") if old_front else None
        self._refresh()
        self.statusBar().showMessage("Front and Back swapped", 3000)

    def _redetect_side(self, side: str) -> None:
        card = self.front if side == "front" else self.back
        if card is None:
            return
        if card.has_manual_correction:
            answer = QMessageBox.question(
                self,
                "Replace manual correction?",
                "Re-detecting will replace the manually adjusted corners. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        result = self._detect_cards([card])[0]
        self._refresh()
        widget = self._card_widget(side)
        if result.success:
            widget.show_corrected()
        else:
            widget.show_original()
        self.statusBar().showMessage(self._detection_message(result), 7000)

    def _adjust_corners(self, side: str) -> None:
        card = self.front if side == "front" else self.back
        if card is None or card.detection_result is None:
            return
        detection_points = card.detection_result.polygon_points
        if len(detection_points) != 4:
            self._show_error(
                "Corners unavailable",
                "Re-detect the card before adjusting its corners.",
            )
            return
        corrector = self.card_processor.perspective_corrector
        try:
            ordered = corrector.order_corners(detection_points)
        except ValueError as exc:
            self._show_error("Corners unavailable", str(exc))
            return
        automatic_points = tuple(
            (float(point[0]), float(point[1])) for point in ordered
        )
        automatic = card.automatic_perspective_result
        if automatic is not None and len(automatic.refined_points) == 4:
            automatic_points = automatic.refined_points
        current_points = (
            card.manual_perspective_result.source_points
            if card.manual_perspective_result is not None
            and len(card.manual_perspective_result.source_points) == 4
            else automatic_points
        )
        dialog = CornerEditorDialog(
            card.original_image,
            automatic_points,
            corrector,
            card.detection_result.confidence,
            current_points=current_points,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.result is None:
            return
        card.apply_manual_correction(dialog.result)
        self._refresh()
        self._card_widget(side).show_corrected()
        self.statusBar().showMessage(
            f"{side.title()} manual correction applied", 5000
        )

    def _reset_correction(self, side: str) -> None:
        card = self.front if side == "front" else self.back
        if card is None:
            return
        self.card_processor.reset_correction(card)
        self._refresh()
        self._card_widget(side).show_corrected()
        self.statusBar().showMessage(
            f"{side.title()} restored to automatic correction", 5000
        )

    def _reset_side(self, side: str) -> None:
        card = self.front if side == "front" else self.back
        if card is None:
            return
        self.card_processor.reset_user_edits(card)
        self._refresh()
        self.statusBar().showMessage(
            f"{side.title()} restored to automatic processing", 5000
        )

    def _set_image_correction(
        self, side: str, state: ImageCorrectionState
    ) -> None:
        card = self.front if side == "front" else self.back
        if card is None:
            return
        card.apply_image_correction(state)
        self._refresh()
        self.statusBar().showMessage(
            f"{side.title()} image correction updated", 3000
        )

    def _show_corrections(self, side: str) -> None:
        card = self.front if side == "front" else self.back
        if card is not None:
            self.preview.select_side(side)  # type: ignore[arg-type]

    def _expand_corrections(self) -> None:
        side = self.preview.selected_side
        if side is None:
            side = "front" if self.front is not None else "back"
        card = self.front if side == "front" else self.back
        if card is not None:
            self.preview.select_side(side)

    def _open_corrections_for_selection(self, side: str) -> None:
        card = self.front if side == "front" else self.back
        if card is None:
            return
        self.corrections_sidebar.set_card(card)
        if not self.corrections_sidebar.isVisible():
            previous = self.splitter.sizes()
            left_width = previous[0] if previous else 390
            self.corrections_sidebar.show()
            sidebar_width = 300
            available = max(0, self.splitter.width() - left_width - sidebar_width)
            self.splitter.setSizes(
                [left_width, max(self.preview.minimumWidth(), available), sidebar_width]
            )
        self.corrections_button.hide()
        self.corrections_sidebar.raise_()
        self.preview.updateGeometry()
        self.preview.update()

    def _collapse_corrections(self) -> None:
        if self.preview.selected_side is not None:
            self.preview.clear_selection(emit=False)
        self.corrections_sidebar.hide()
        self.corrections_button.show()
        self.preview.updateGeometry()
        self.preview.update()

    def _select_correction_preset(self, category: str, key: str) -> None:
        side = self.preview.selected_side
        if side is None:
            return
        state = self.corrections_sidebar.selected_state(category, key)
        if state is not None:
            self._set_image_correction(side, state)

    def _reset_image_corrections(self) -> None:
        side = self.preview.selected_side
        if side is not None:
            self._set_image_correction(side, ImageCorrectionState())

    def _reset_detection(self, side: str) -> None:
        card = self.front if side == "front" else self.back
        if card is None:
            return
        self.card_processor.reset(card)
        self._refresh()
        self._card_widget(side).show_original()
        self.statusBar().showMessage(
            f"{side.title()} restored to the original image", 5000
        )

    def _detect_cards(
        self, cards: list[CardSide]
    ) -> list[CardDetectionResult]:
        results: list[CardDetectionResult] = []
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for card in cards:
                try:
                    results.append(self.card_processor.detect(card))
                except Exception:
                    failure = CardDetectionResult(
                        success=False,
                        confidence=0.0,
                        confidence_level="none",
                        method="processing_error",
                        debug_info={"reason": "Detection processing failed"},
                    )
                    card.apply_detection(failure)
                    results.append(failure)
        finally:
            QApplication.restoreOverrideCursor()
        return results

    @staticmethod
    def _detection_message(result: CardDetectionResult) -> str:
        if result.success and result.confidence_level == "high":
            return "Card detected automatically."
        if result.success:
            return "Card detected; review recommended."
        return "Card detection uncertain; using the original image."

    def _card_widget(self, side: str) -> CardInputWidget:
        return self.front_widget if side == "front" else self.back_widget

    def _adjust_position(self, side: str, delta_mm: float) -> None:
        offset = self.engine.adjust_vertical_offset(side, delta_mm)  # type: ignore[arg-type]
        self._refresh()
        direction = "down" if delta_mm > 0 else "up"
        self.statusBar().showMessage(
            f"{side.title()} moved {direction}; offset is {offset:+g} mm",
            3000,
        )

    def _reset_position(self, side: str) -> None:
        self.engine.reset_vertical_offset(side)  # type: ignore[arg-type]
        self._refresh()
        self.statusBar().showMessage(f"{side.title()} position reset", 3000)

    def _refresh(self) -> None:
        self.front_widget.set_card(self.front)
        self.back_widget.set_card(self.back)
        self.corrections_button.setEnabled(
            self.front is not None or self.back is not None
        )
        self.preview.set_position_offsets(
            self.engine.vertical_offset("front"),
            self.engine.vertical_offset("back"),
        )
        self.preview.set_sides(self.front, self.back)
        selected = self.preview.selected_side
        if self.corrections_sidebar.isVisible() and selected is not None:
            card = self.front if selected == "front" else self.back
            if card is not None:
                self.corrections_sidebar.set_card(card)

    def _export_pdf(self) -> None:
        try:
            path = self._automatic_export_path(".pdf")
        except OSError as exc:
            self._show_error("Export failed", f"Could not access Downloads: {exc}")
            return
        self._run_export("PDF", path, self.pdf_exporter.export)

    def _export_jpg(self) -> None:
        try:
            path = self._automatic_export_path(".jpg")
        except OSError as exc:
            self._show_error("Export failed", f"Could not access Downloads: {exc}")
            return
        self._run_export("JPG", path, self.jpg_exporter.export)

    @staticmethod
    def _downloads_directory() -> Path:
        standard_path = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DownloadLocation
        )
        return Path(standard_path) if standard_path else Path.home() / "Downloads"

    def _automatic_export_path(self, suffix: str) -> Path:
        downloads = self._downloads_directory()
        downloads.mkdir(parents=True, exist_ok=True)
        candidate = downloads / f"card-layout{suffix}"
        sequence = 1
        while candidate.exists():
            candidate = downloads / f"card-layout ({sequence}){suffix}"
            sequence += 1
        return candidate

    def _run_export(self, kind: str, path: str | Path, exporter: object) -> None:
        exported = False
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            exporter(path, self.front, self.back)  # type: ignore[operator]
            self.statusBar().showMessage(f"{kind} exported to {path}", 8000)
            exported = True
        except ExportError as exc:
            self._show_error("Export failed", str(exc))
        finally:
            QApplication.restoreOverrideCursor()
        if exported:
            output_folder = Path(path).resolve().parent
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_folder)))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
        if paths and len(paths) <= 2 and all(p.suffix.lower() in SUPPORTED_SUFFIXES for p in paths):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()][:2]
        if len(paths) == 1 and paths[0].suffix.lower() == ".pdf":
            self._load_two_page_pdf(str(paths[0]))
        elif paths:
            self._load_side(str(paths[0]), "front")
            if len(paths) > 1:
                self._load_side(str(paths[1]), "back")
        event.acceptProposedAction()

    def _show_error(self, title: str, message: str) -> None:
        self.statusBar().showMessage(message, 7000)
        QMessageBox.warning(self, title, message)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f4f6f9; color: #1e293b; font-family: "Segoe UI"; font-size: 10pt; }
            QFrame#header, QFrame#exportBar { background: #ffffff; border: 0; }
            QFrame#header { border-bottom: 1px solid #dce2ea; }
            QFrame#exportBar { border-top: 1px solid #dce2ea; }
            QLabel#appTitle { font-size: 18pt; font-weight: 700; color: #0f172a; }
            QLabel#subtitle { color: #64748b; font-size: 9pt; }
            QLabel#sectionCaption, QLabel#sideTitle { color: #475569; font-size: 9pt; font-weight: 700; letter-spacing: 1px; }
            QLabel#fileName { color: #0f172a; font-weight: 600; }
            QLabel#detectionStatus { font-size: 8pt; }
            QFrame#correctionsPanel { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 9px; }
            QLabel#correctionsTitle { color: #0f172a; font-size: 11pt; font-weight: 700; border: 0; }
            QLabel#correctionsHint { color: #64748b; font-size: 8pt; border: 0; }
            QLabel#correctionsCategory { color: #475569; font-size: 8pt; font-weight: 700; border: 0; margin-top: 4px; }
            QToolButton { background: #ffffff; color: #475569; border: 1px solid #d8e0ea; border-radius: 6px; padding: 4px; font-size: 8pt; }
            QToolButton:hover { background: #f1f5fb; border-color: #9fb4d3; }
            QToolButton:checked { background: #e8f0ff; color: #194b9b; border: 2px solid #4f7fc8; }
            QFrame#previewControls { background: #ffffff; border: 1px solid #b8c5d7; border-radius: 9px; }
            QLabel#previewSelection { color: #194b9b; font-size: 9pt; font-weight: 700; border: 0; }
            QPushButton#previewControlButton { background: #edf3fb; color: #234c86; border: 1px solid #bdcce0; padding: 0 10px; }
            QPushButton#previewControlButton:hover { background: #dfeafb; }
            QPushButton#previewDoneButton { background: transparent; color: #64748b; border: 0; padding: 0 8px; }
            QPushButton#compactExpandButton { min-height: 28px; padding: 0 10px; background: #ffffff; color: #234c86; border: 1px solid #bdc9d9; font-size: 8pt; }
            QPushButton#compactExpandButton:hover { background: #edf3fb; }
            QPushButton#swapButton { min-height: 28px; padding: 0 10px; background: transparent; color: #526176; border: 1px solid #d5dce6; font-size: 8pt; }
            QPushButton#swapButton:hover { background: #edf3fb; color: #234c86; }
            QPushButton#sidebarCloseButton { min-width: 0; min-height: 0; padding: 0; background: transparent; color: #64748b; border: 0; font-size: 15pt; font-weight: 400; }
            QPushButton#sidebarCloseButton:hover { background: #eef2f7; color: #0f172a; }
            QPushButton#sidebarResetButton { min-height: 30px; background: #ffffff; color: #526176; border: 1px solid #cbd4e0; font-size: 8pt; }
            QPushButton#sidebarResetButton:hover { background: #edf3fb; }
            QLabel#presetValue { background: #e8f0ff; color: #194b9b; border: 1px solid #c8d9f5; border-radius: 7px; padding: 10px; font-weight: 600; }
            QLabel#privacyNote { background: #ecfdf5; color: #166534; border-radius: 7px; padding: 10px; }
            QFrame#cardInput { background: #ffffff; border: 1px solid #dce2ea; border-radius: 9px; }
            QLabel#sidePreview { background: #f7f9fc; color: #8a96a8; border: 1px dashed #b9c3d1; border-radius: 5px; }
            QWidget#previewPanel { background: #e9edf3; }
            QPushButton { min-height: 34px; padding: 0 14px; border-radius: 6px; font-weight: 600; }
            QPushButton#primaryButton { background: #245fc5; color: white; border: 1px solid #245fc5; }
            QPushButton#primaryButton:hover { background: #1d4fa8; }
            QPushButton#secondaryButton { background: #ffffff; color: #234c86; border: 1px solid #aebdd1; }
            QPushButton#secondaryButton:hover { background: #edf3fb; }
            QPushButton#quietButton { background: transparent; color: #64748b; border: 1px solid #d5dce6; }
            QPushButton#detectionButton { min-height: 29px; padding: 0 7px; background: #ffffff; color: #526176; border: 1px solid #cbd4e0; font-size: 8pt; }
            QPushButton#detectionButton:checked { background: #e8f0ff; color: #194b9b; border-color: #83a7df; }
            QPushButton#detectionButton:hover { background: #edf3fb; }
            QPushButton:disabled { color: #aab2bf; background: #f1f3f6; border-color: #e2e6ec; }
            QScrollArea { background: #f4f6f9; }
            QSplitter::handle { background: #dce2ea; width: 1px; }
            QStatusBar { background: #ffffff; color: #64748b; border-top: 1px solid #dce2ea; }
            """
        )
