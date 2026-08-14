from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from cardlayout.models.card_side import CardSide, SideName
from cardlayout.ui.image_utils import pil_to_qimage


class CardInputWidget(QFrame):
    choose_requested = Signal(str)
    clear_requested = Signal(str)
    detect_requested = Signal(str)
    reset_detection_requested = Signal(str)
    adjust_corners_requested = Signal(str)
    reset_correction_requested = Signal(str)

    def __init__(self, side: SideName, parent: object | None = None) -> None:
        super().__init__(parent)  # type: ignore[arg-type]
        self.side = side
        self._card: CardSide | None = None
        self._preview_mode = "original"
        self._source_pixmap: QPixmap | None = None
        self.setObjectName("cardInput")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        title = QLabel(side.upper())
        title.setObjectName("sideTitle")
        self.detection_status = QLabel("Not processed")
        self.detection_status.setObjectName("detectionStatus")
        heading = QHBoxLayout()
        heading.addWidget(title)
        heading.addStretch()
        heading.addWidget(self.detection_status)
        self.file_label = QLabel("No file selected")
        self.file_label.setObjectName("fileName")
        self.file_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.preview = QLabel(f"{side.title()} preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(115)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview.setObjectName("sidePreview")

        choose = QPushButton(f"Choose {side.title()}")
        choose.setObjectName("secondaryButton")
        choose.clicked.connect(lambda: self.choose_requested.emit(self.side))
        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("quietButton")
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(lambda: self.clear_requested.emit(self.side))

        buttons = QHBoxLayout()
        buttons.addWidget(choose, 1)
        buttons.addWidget(self.clear_button)

        self.original_button = QPushButton("Original")
        self.original_button.setObjectName("detectionButton")
        self.original_button.setCheckable(True)
        self.original_button.clicked.connect(lambda: self._set_preview_mode("original"))
        self.detected_button = QPushButton("Detected")
        self.detected_button.setObjectName("detectionButton")
        self.detected_button.setCheckable(True)
        self.detected_button.clicked.connect(lambda: self._set_preview_mode("detected"))
        self.corrected_button = QPushButton("Corrected")
        self.corrected_button.setObjectName("detectionButton")
        self.corrected_button.setCheckable(True)
        self.corrected_button.clicked.connect(lambda: self._set_preview_mode("corrected"))
        self.redetect_button = QPushButton("Re-detect")
        self.redetect_button.setObjectName("detectionButton")
        self.redetect_button.clicked.connect(lambda: self.detect_requested.emit(self.side))
        self.adjust_corners_button = QPushButton("Adjust Corners")
        self.adjust_corners_button.setObjectName("detectionButton")
        self.adjust_corners_button.clicked.connect(
            lambda: self.adjust_corners_requested.emit(self.side)
        )
        self.reset_correction_button = QPushButton("Reset Correction")
        self.reset_correction_button.setObjectName("detectionButton")
        self.reset_correction_button.clicked.connect(
            lambda: self.reset_correction_requested.emit(self.side)
        )
        self.reset_detection_button = QPushButton("Reset Detection")
        self.reset_detection_button.setObjectName("detectionButton")
        self.reset_detection_button.setToolTip("Reset detection and use the original image")
        self.reset_detection_button.clicked.connect(
            lambda: self.reset_detection_requested.emit(self.side)
        )
        detection_buttons = QHBoxLayout()
        detection_buttons.setSpacing(5)
        detection_buttons.addWidget(self.original_button)
        detection_buttons.addWidget(self.detected_button)
        detection_buttons.addWidget(self.corrected_button)
        detection_buttons.addStretch()

        correction_buttons = QHBoxLayout()
        correction_buttons.setSpacing(5)
        detection_buttons.addWidget(self.redetect_button)
        correction_buttons.addWidget(self.adjust_corners_button)
        correction_buttons.addWidget(self.reset_correction_button)
        correction_buttons.addWidget(self.reset_detection_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(9)
        layout.addLayout(heading)
        layout.addWidget(self.file_label)
        layout.addWidget(self.preview, 1)
        layout.addLayout(buttons)
        layout.addLayout(detection_buttons)
        layout.addLayout(correction_buttons)

    def set_card(self, card: CardSide | None) -> None:
        is_new_card = card is not self._card
        self._card = card
        if card is None:
            self._preview_mode = "original"
            self._source_pixmap = None
            self.file_label.setText("No file selected")
            self.file_label.setToolTip("")
            self.preview.setPixmap(QPixmap())
            self.preview.setText(f"{self.side.title()} preview")
            self.clear_button.setEnabled(False)
            self._update_detection_controls()
            return

        if is_new_card:
            if card.rectified_image is not None:
                self._preview_mode = "corrected"
            else:
                self._preview_mode = "detected" if card.detected_image is not None else "original"
        elif self._preview_mode == "detected" and card.detected_image is None:
            self._preview_mode = "original"
        elif self._preview_mode == "corrected" and card.rectified_image is None:
            self._preview_mode = "detected" if card.detected_image is not None else "original"
        self.file_label.setText(card.display_name)
        self.file_label.setToolTip(str(card.source_path))
        self.preview.setText("")
        self.clear_button.setEnabled(True)
        self._update_detection_controls()
        self._update_preview_source()

    def show_detected(self) -> None:
        if self._card is not None and self._card.detected_image is not None:
            self._set_preview_mode("detected")

    def show_original(self) -> None:
        self._set_preview_mode("original")

    def show_corrected(self) -> None:
        if self._card is not None and self._card.rectified_image is not None:
            self._set_preview_mode("corrected")
        else:
            self.show_detected()

    def _set_preview_mode(self, mode: str) -> None:
        if self._card is None:
            return
        if mode == "detected" and self._card.detected_image is None:
            return
        if mode == "corrected" and self._card.rectified_image is None:
            return
        self._preview_mode = mode
        self._update_detection_controls()
        self._update_preview_source()

    def _update_preview_source(self) -> None:
        if self._card is None:
            return
        if self._preview_mode == "corrected" and self._card.rectified_image is not None:
            image = self._card.rectified_image
        elif self._preview_mode == "detected" and self._card.detected_image is not None:
            image = self._card.detected_image
        else:
            image = self._card.original_image
        self._source_pixmap = QPixmap.fromImage(pil_to_qimage(image))
        self._update_pixmap()

    def _update_detection_controls(self) -> None:
        has_card = self._card is not None
        has_detection = has_card and self._card.detected_image is not None
        has_correction = has_card and self._card.rectified_image is not None
        self.original_button.setEnabled(has_card)
        self.detected_button.setEnabled(has_detection)
        self.corrected_button.setEnabled(has_correction)
        self.redetect_button.setEnabled(has_card)
        self.adjust_corners_button.setEnabled(
            has_card
            and self._card.detection_result is not None
            and len(self._card.detection_result.polygon_points) == 4
        )
        self.reset_correction_button.setEnabled(
            has_card and self._card.has_manual_correction
        )
        self.reset_detection_button.setEnabled(
            has_card and self._card.detection_result is not None
        )
        self.original_button.setChecked(has_card and self._preview_mode == "original")
        self.detected_button.setChecked(has_detection and self._preview_mode == "detected")
        self.corrected_button.setChecked(
            has_correction and self._preview_mode == "corrected"
        )

        result = self._card.detection_result if self._card is not None else None
        correction = self._card.active_perspective_result if self._card is not None else None
        if correction is not None and correction.success:
            self.detection_status.setText(correction.status_text)
            color = "#16803b" if correction.status == "corrected" or correction.method == "manual" else "#a15c00"
        elif correction is not None:
            self.detection_status.setText("Correction failed")
            color = "#b42318"
        elif result is None:
            self.detection_status.setText("Not processed")
            color = "#64748b"
        elif result.success and result.confidence_level == "high":
            self.detection_status.setText(result.status_text)
            color = "#16803b"
        elif result.success:
            self.detection_status.setText(result.status_text)
            color = "#a15c00"
        else:
            self.detection_status.setText(result.status_text)
            color = "#b42318"
        self.detection_status.setStyleSheet(f"color: {color}; font-weight: 600;")

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)  # type: ignore[arg-type]
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._source_pixmap is None or self.preview.width() < 2:
            return
        target = self.preview.size()
        target.setWidth(max(1, target.width() - 16))
        target.setHeight(max(1, target.height() - 16))
        self.preview.setPixmap(
            self._source_pixmap.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
