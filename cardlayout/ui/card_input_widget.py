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

    def __init__(self, side: SideName, parent: object | None = None) -> None:
        super().__init__(parent)  # type: ignore[arg-type]
        self.side = side
        self._source_pixmap: QPixmap | None = None
        self.setObjectName("cardInput")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        title = QLabel(side.upper())
        title.setObjectName("sideTitle")
        self.file_label = QLabel("No file selected")
        self.file_label.setObjectName("fileName")
        self.file_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.preview = QLabel(f"{side.title()} preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(145)
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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(9)
        layout.addWidget(title)
        layout.addWidget(self.file_label)
        layout.addWidget(self.preview, 1)
        layout.addLayout(buttons)

    def set_card(self, card: CardSide | None) -> None:
        if card is None:
            self._source_pixmap = None
            self.file_label.setText("No file selected")
            self.file_label.setToolTip("")
            self.preview.setPixmap(QPixmap())
            self.preview.setText(f"{self.side.title()} preview")
            self.clear_button.setEnabled(False)
            return

        self.file_label.setText(card.display_name)
        self.file_label.setToolTip(str(card.source_path))
        self._source_pixmap = QPixmap.fromImage(pil_to_qimage(card.processed_image))
        self.preview.setText("")
        self.clear_button.setEnabled(True)
        self._update_pixmap()

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
