from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
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
from cardlayout.models.layout import A4_LAYOUT
from cardlayout.services.card_detector import CardDetector
from cardlayout.services.card_processing import CardProcessingService
from cardlayout.services.image_exporter import ExportError, ImageExporter
from cardlayout.services.input_loader import InputLoadError, InputLoader, SUPPORTED_SUFFIXES
from cardlayout.services.layout_engine import LayoutEngine
from cardlayout.services.page_renderer import PageRenderer
from cardlayout.services.pdf_exporter import PDFExporter
from cardlayout.ui.card_input_widget import CardInputWidget
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
        subtitle = QLabel("Arrange card fronts and backs for accurate A4 printing")
        subtitle.setObjectName("subtitle")

        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        pdf_button = QPushButton("Open 2-Page PDF")
        pdf_button.setObjectName("secondaryButton")
        pdf_button.clicked.connect(self._choose_two_page_pdf)
        swap_button = QPushButton("Swap Front / Back")
        swap_button.setObjectName("secondaryButton")
        swap_button.clicked.connect(self._swap_sides)

        header = QHBoxLayout()
        header.setContentsMargins(22, 15, 22, 15)
        header.addLayout(title_box)
        header.addStretch()
        header.addWidget(pdf_button)
        header.addWidget(swap_button)

        header_widget = QFrame()
        header_widget.setObjectName("header")
        header_widget.setLayout(header)

        self.front_widget = CardInputWidget("front")
        self.back_widget = CardInputWidget("back")
        for widget in (self.front_widget, self.back_widget):
            widget.choose_requested.connect(self._choose_side)
            widget.clear_requested.connect(self._clear_side)
            widget.detect_requested.connect(self._redetect_side)
            widget.reset_detection_requested.connect(self._reset_detection)
            widget.adjust_corners_requested.connect(self._adjust_corners)
            widget.reset_correction_requested.connect(self._reset_correction)

        size_caption = QLabel("CARD SIZE")
        size_caption.setObjectName("sectionCaption")
        size_value = QLabel(MALAYSIA_IC.label)
        size_value.setObjectName("presetValue")
        size_value.setWordWrap(True)
        local_note = QLabel("Detection and export run locally. No uploads or OCR.")
        local_note.setWordWrap(True)
        local_note.setObjectName("privacyNote")

        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(18, 18, 18, 18)
        controls_layout.setSpacing(13)
        controls_layout.addWidget(self.front_widget)
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

        preview_title = QLabel("A4 PORTRAIT PREVIEW")
        preview_title.setObjectName("sectionCaption")
        preview_note = QLabel("Click Front or Back to adjust its vertical position")
        preview_note.setObjectName("subtitle")
        self.preview = PagePreview(self.engine)
        self.preview.position_adjust_requested.connect(self._adjust_position)
        self.preview.position_reset_requested.connect(self._reset_position)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        preview_heading = QHBoxLayout()
        preview_heading.addWidget(preview_title)
        preview_heading.addStretch()
        preview_heading.addWidget(preview_note)
        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(18, 18, 18, 18)
        preview_layout.addLayout(preview_heading)
        preview_layout.addWidget(self.preview, 1)
        preview_panel = QWidget()
        preview_panel.setObjectName("previewPanel")
        preview_panel.setLayout(preview_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(preview_panel)
        splitter.setSizes([390, 850])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

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
        central_layout.addWidget(splitter, 1)
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
        self.preview.set_position_offsets(
            self.engine.vertical_offset("front"),
            self.engine.vertical_offset("back"),
        )
        self.preview.set_sides(self.front, self.back)

    def _export_pdf(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export A4 PDF", "card-layout.pdf", "PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        self._run_export("PDF", path, self.pdf_exporter.export)

    def _export_jpg(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export A4 JPG", "card-layout.jpg", "JPEG (*.jpg *.jpeg)")
        if not path:
            return
        if not path.lower().endswith((".jpg", ".jpeg")):
            path += ".jpg"
        self._run_export("JPG", path, self.jpg_exporter.export)

    def _run_export(self, kind: str, path: str, exporter: object) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            exporter(path, self.front, self.back)  # type: ignore[operator]
            self.statusBar().showMessage(f"{kind} exported to {path}", 8000)
            QMessageBox.information(self, "Export complete", f"The A4 {kind} was saved successfully.")
        except ExportError as exc:
            self._show_error("Export failed", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

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
            QFrame#previewControls { background: #ffffff; border: 1px solid #b8c5d7; border-radius: 9px; }
            QLabel#previewSelection { color: #194b9b; font-size: 9pt; font-weight: 700; border: 0; }
            QPushButton#previewControlButton { background: #edf3fb; color: #234c86; border: 1px solid #bdcce0; padding: 0 10px; }
            QPushButton#previewControlButton:hover { background: #dfeafb; }
            QPushButton#previewDoneButton { background: transparent; color: #64748b; border: 0; padding: 0 8px; }
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
