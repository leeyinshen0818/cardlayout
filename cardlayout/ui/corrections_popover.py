from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
)

from cardlayout.models.card_side import CardSide
from cardlayout.models.image_correction import (
    ImageCorrectionState,
    SHARPEN_PRESETS,
    TONE_PRESETS,
)
from cardlayout.services.image_corrections import correction_thumbnail
from cardlayout.ui.image_utils import pil_to_qimage


class CorrectionsPopover(QFrame):
    """Compact thumbnail correction panel usable as an embedded sidebar."""

    preset_selected = Signal(str, str)
    orientation_requested = Signal(str)
    collapse_requested = Signal()
    reset_requested = Signal()

    def __init__(
        self,
        parent: object | None = None,
        *,
        popup: bool = True,
        columns: int | None = None,
        show_close: bool = False,
    ) -> None:
        window_type = Qt.WindowType.Popup if popup else Qt.WindowType.Widget
        super().__init__(parent, window_type)  # type: ignore[arg-type]
        self.setObjectName("correctionsPanel")
        self._card: CardSide | None = None
        self._buttons: dict[tuple[str, str], QToolButton] = {}
        self._thumbnail_cache: dict[tuple[int, str, str], QIcon] = {}
        self._columns = columns or 4

        self.title = QLabel("Corrections")
        self.title.setObjectName("correctionsTitle")
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(6)
        title_row.addWidget(self.title, 1)
        self.close_button = QPushButton("×")
        self.close_button.setObjectName("sidebarCloseButton")
        self.close_button.setToolTip("Collapse corrections sidebar")
        self.close_button.setFixedSize(28, 28)
        self.close_button.clicked.connect(self.collapse_requested.emit)
        self.close_button.setVisible(show_close)
        title_row.addWidget(self.close_button)
        hint = QLabel("Changes are applied from the corrected card image.")
        hint.setObjectName("correctionsHint")
        hint.setWordWrap(True)

        self.content_layout = QVBoxLayout(self)
        self.content_layout.setContentsMargins(12, 10, 12, 12)
        self.content_layout.setSpacing(6)
        self.content_layout.addLayout(title_row)
        self.content_layout.addWidget(hint)
        orientation_label = QLabel("Orientation")
        orientation_label.setObjectName("correctionsCategory")
        self.content_layout.addWidget(orientation_label)
        self.orientation_status = QLabel("Current: Normal")
        self.orientation_status.setObjectName("orientationStatus")
        self.content_layout.addWidget(self.orientation_status)
        orientation_grid = QGridLayout()
        orientation_grid.setHorizontalSpacing(6)
        orientation_grid.setVerticalSpacing(5)
        self.orientation_buttons: dict[str, QPushButton] = {}
        for index, (action, text) in enumerate(
            (
                ("flip_horizontal", "Flip Horizontal"),
                ("flip_vertical", "Flip Vertical"),
                ("rotate_180", "Rotate 180°"),
                ("reset", "Reset Orientation"),
            )
        ):
            button = QPushButton(text)
            button.setObjectName("orientationButton")
            button.clicked.connect(
                lambda checked=False, selected=action: self.orientation_requested.emit(
                    selected
                )
            )
            self.orientation_buttons[action] = button
            orientation_grid.addWidget(button, index // 2, index % 2)
        self.content_layout.addLayout(orientation_grid)
        self._add_category(
            self.content_layout,
            "Sharpen / Soften",
            "sharpen",
            [(key, preset.label) for key, preset in SHARPEN_PRESETS.items()],
            columns=columns or 4,
        )
        self._add_category(
            self.content_layout,
            "Brightness / Contrast",
            "tone",
            [(key, preset.label) for key, preset in TONE_PRESETS.items()],
            columns=columns or 3,
        )
        self.reset_button = QPushButton("Reset corrections")
        self.reset_button.setObjectName("sidebarResetButton")
        self.reset_button.setToolTip(
            "Restore sharpening, brightness, and contrast to Normal"
        )
        self.reset_button.clicked.connect(self.reset_requested.emit)
        self.content_layout.addWidget(self.reset_button)

    def set_compact(self, compact: bool) -> None:
        """Reduce sidebar density on smaller logical desktop resolutions."""
        if compact:
            self.content_layout.setContentsMargins(9, 7, 9, 9)
            self.content_layout.setSpacing(4)
            self.close_button.setFixedSize(24, 24)
            icon_size = QSize(68, 40)
            button_size = QSize(104, 66)
            orientation_height = 24
        else:
            self.content_layout.setContentsMargins(12, 10, 12, 12)
            self.content_layout.setSpacing(6)
            self.close_button.setFixedSize(28, 28)
            icon_size = QSize(76, 46) if self._columns <= 2 else QSize(88, 56)
            button_size = QSize(122, 76) if self._columns <= 2 else QSize(104, 88)
            orientation_height = 28
        for button in self._buttons.values():
            button.setIconSize(icon_size)
            button.setFixedSize(button_size)
        for button in self.orientation_buttons.values():
            button.setFixedHeight(orientation_height)

    def _add_category(
        self,
        parent_layout: QVBoxLayout,
        heading: str,
        category: str,
        presets: list[tuple[str, str]],
        *,
        columns: int,
    ) -> None:
        label = QLabel(heading)
        label.setObjectName("correctionsCategory")
        parent_layout.addWidget(label)
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        compact = columns <= 2
        for index, (key, text) in enumerate(presets):
            button = QToolButton()
            button.setObjectName(f"correction_{category}_{key}")
            display_text = (
                "Strong"
                if compact and key == "strong_bright_contrast"
                else text
            )
            button.setText(display_text)
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setIconSize(QSize(76, 46) if compact else QSize(88, 56))
            if compact:
                button.setFixedSize(122, 76)
            else:
                button.setFixedSize(104, 88)
            button.setToolTip(text)
            button.clicked.connect(
                lambda checked=False, group=category, preset=key: self.preset_selected.emit(
                    group, preset
                )
            )
            self._buttons[(category, key)] = button
            grid.addWidget(button, index // columns, index % columns)
        parent_layout.addLayout(grid)

    def set_card(self, card: CardSide) -> None:
        self._card = card
        self.title.setText(f"Corrections · {card.side.upper()}")
        state = card.image_correction_state
        orientation = card.orientation_state
        self.orientation_status.setText(f"Current: {orientation.label}")
        self.orientation_buttons["reset"].setEnabled(not orientation.is_normal)
        source = card.orientation_image
        for (category, key), button in self._buttons.items():
            preview_state = (
                state.with_sharpen(key)  # type: ignore[arg-type]
                if category == "sharpen"
                else state.with_tone(key)  # type: ignore[arg-type]
            )
            cache_key = (id(source), preview_state.sharpen, preview_state.tone)
            icon = self._thumbnail_cache.get(cache_key)
            if icon is None:
                thumbnail = correction_thumbnail(source, preview_state)
                icon = QIcon(QPixmap.fromImage(pil_to_qimage(thumbnail)))
                self._thumbnail_cache[cache_key] = icon
                if len(self._thumbnail_cache) > 80:
                    self._thumbnail_cache = {cache_key: icon}
            button.setIcon(icon)
            button.setChecked(
                key == (state.sharpen if category == "sharpen" else state.tone)
            )
        self.reset_button.setEnabled(not state.is_normal)

    def selected_state(self, category: str, key: str) -> ImageCorrectionState | None:
        if self._card is None:
            return None
        state = self._card.image_correction_state
        if category == "sharpen":
            return state.with_sharpen(key)  # type: ignore[arg-type]
        if category == "tone":
            return state.with_tone(key)  # type: ignore[arg-type]
        return None
