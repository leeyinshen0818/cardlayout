from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent, QPixmap
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


class ClickablePreview(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self.clicked.emit()
        super().mousePressEvent(event)


class CardInputWidget(QFrame):
    choose_requested = Signal(str)
    clear_requested = Signal(str)
    adjust_corners_requested = Signal(str)
    reset_requested = Signal(str)
    corrections_requested = Signal(str)

    def __init__(self, side: SideName, parent: object | None = None) -> None:
        super().__init__(parent)  # type: ignore[arg-type]
        self.side = side
        self._card: CardSide | None = None
        self._source_pixmap: QPixmap | None = None
        self.setObjectName("cardInput")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        title = QLabel(side.upper())
        title.setObjectName("sideTitle")
        self.detection_status = QLabel("Ready")
        self.detection_status.setObjectName("detectionStatus")
        heading = QHBoxLayout()
        heading.addWidget(title)
        heading.addStretch()
        heading.addWidget(self.detection_status)

        self.file_label = QLabel("No file selected")
        self.file_label.setObjectName("fileName")
        self.file_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.file_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.preview = ClickablePreview(f"{side.title()} preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(115)
        self.preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.preview.setObjectName("sidePreview")
        self.preview.setToolTip("Click for image corrections")
        self.preview.clicked.connect(
            lambda: self.corrections_requested.emit(self.side)
        )

        self.choose_button = QPushButton(f"Choose {side.title()}")
        self.choose_button.setObjectName("secondaryButton")
        self.choose_button.clicked.connect(
            lambda: self.choose_requested.emit(self.side)
        )
        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("quietButton")
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(lambda: self.clear_requested.emit(self.side))

        file_buttons = QHBoxLayout()
        file_buttons.addWidget(self.choose_button, 1)
        file_buttons.addWidget(self.clear_button)

        self.adjust_corners_button = QPushButton("Adjust Corners")
        self.adjust_corners_button.setObjectName("detectionButton")
        self.adjust_corners_button.clicked.connect(
            lambda: self.adjust_corners_requested.emit(self.side)
        )
        self.reset_button = QPushButton("Reset")
        self.reset_button.setObjectName("detectionButton")
        self.reset_button.setToolTip(
            "Restore automatic corners and Normal image corrections"
        )
        self.reset_button.clicked.connect(lambda: self.reset_requested.emit(self.side))

        self.edit_buttons = QHBoxLayout()
        self.edit_buttons.setSpacing(7)
        self.edit_buttons.addWidget(self.adjust_corners_button, 1)
        self.edit_buttons.addWidget(self.reset_button, 1)

        self.content_layout = QVBoxLayout(self)
        self.content_layout.setContentsMargins(16, 14, 16, 14)
        self.content_layout.setSpacing(9)
        self.content_layout.addLayout(heading)
        self.content_layout.addWidget(self.file_label)
        self.content_layout.addWidget(self.preview, 1)
        self.content_layout.addLayout(file_buttons)
        self.content_layout.addLayout(self.edit_buttons)

    def set_compact(self, compact: bool) -> None:
        """Adjust only UI density; card state and processing remain untouched."""
        if compact:
            self.preview.setMinimumHeight(88)
            self.content_layout.setContentsMargins(11, 9, 11, 9)
            self.content_layout.setSpacing(6)
            self.edit_buttons.setSpacing(5)
        else:
            self.preview.setMinimumHeight(115)
            self.content_layout.setContentsMargins(16, 14, 16, 14)
            self.content_layout.setSpacing(9)
            self.edit_buttons.setSpacing(7)

    def set_card(self, card: CardSide | None) -> None:
        self._card = card
        if card is None:
            self._source_pixmap = None
            self.file_label.setText("No file selected")
            self.file_label.setToolTip("")
            self.preview.setPixmap(QPixmap())
            self.preview.setText(f"{self.side.title()} preview")
            self.preview.setEnabled(False)
            self.clear_button.setEnabled(False)
            self._update_controls()
            return

        self.file_label.setText(card.display_name)
        self.file_label.setToolTip(str(card.source_path))
        self.preview.setText("")
        self.preview.setEnabled(True)
        self.clear_button.setEnabled(True)
        self._update_controls()
        self._update_preview_source()

    def show_detected(self) -> None:
        self._update_preview_source()

    def show_original(self) -> None:
        self._update_preview_source()

    def show_corrected(self) -> None:
        self._update_preview_source()

    def _update_preview_source(self) -> None:
        if self._card is None:
            return
        self._source_pixmap = QPixmap.fromImage(pil_to_qimage(self._card.best_image))
        self._update_pixmap()

    def _update_controls(self) -> None:
        card = self._card
        has_card = card is not None
        self.adjust_corners_button.setEnabled(
            has_card
            and card.detection_result is not None
            and len(card.detection_result.polygon_points) == 4
        )
        self.reset_button.setEnabled(
            has_card
            and (
                card.has_manual_correction
                or not card.image_correction_state.is_normal
            )
        )

        if card is None:
            text, color = "Ready", "#64748b"
        elif card.has_manual_correction:
            text, color = "Ready", "#16803b"
        elif not card.image_correction_state.is_normal:
            text, color = "Corrected", "#16803b"
        else:
            correction = card.automatic_perspective_result
            detection = card.detection_result
            if correction is not None and correction.success:
                if correction.status == "corrected":
                    text, color = "Corrected", "#16803b"
                else:
                    text, color = "Review corners", "#a15c00"
            elif detection is not None and detection.success:
                text, color = "Review corners", "#a15c00"
            elif detection is not None:
                text, color = "Review corners", "#b42318"
            else:
                text, color = "Ready", "#64748b"
        self.detection_status.setText(text)
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
