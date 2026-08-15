from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QToolButton, QVBoxLayout

from cardlayout.models.card_side import CardSide
from cardlayout.models.image_correction import (
    ImageCorrectionState,
    SHARPEN_PRESETS,
    TONE_PRESETS,
)
from cardlayout.services.image_corrections import correction_thumbnail
from cardlayout.ui.image_utils import pil_to_qimage


class CorrectionsPopover(QFrame):
    """Thumbnail correction panel usable as a popup or embedded preview overlay."""

    preset_selected = Signal(str, str)

    def __init__(
        self,
        parent: object | None = None,
        *,
        popup: bool = True,
        columns: int | None = None,
    ) -> None:
        window_type = Qt.WindowType.Popup if popup else Qt.WindowType.Widget
        super().__init__(parent, window_type)  # type: ignore[arg-type]
        self.setObjectName("correctionsPanel")
        self._card: CardSide | None = None
        self._buttons: dict[tuple[str, str], QToolButton] = {}
        self._thumbnail_cache: dict[tuple[int, str, str], QIcon] = {}

        self.title = QLabel("Corrections")
        self.title.setObjectName("correctionsTitle")
        hint = QLabel("Changes are applied from the corrected card image.")
        hint.setObjectName("correctionsHint")
        hint.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(8)
        layout.addWidget(self.title)
        layout.addWidget(hint)
        self._add_category(
            layout,
            "Sharpen / Soften",
            "sharpen",
            [(key, preset.label) for key, preset in SHARPEN_PRESETS.items()],
            columns=columns or 4,
        )
        self._add_category(
            layout,
            "Brightness / Contrast",
            "tone",
            [(key, preset.label) for key, preset in TONE_PRESETS.items()],
            columns=columns or 3,
        )

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
        grid.setHorizontalSpacing(7)
        grid.setVerticalSpacing(7)
        for index, (key, text) in enumerate(presets):
            button = QToolButton()
            button.setObjectName(f"correction_{category}_{key}")
            button.setText(text)
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setIconSize(QSize(88, 56))
            button.setFixedSize(104, 88)
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
        source = card.geometry_image
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

    def selected_state(self, category: str, key: str) -> ImageCorrectionState | None:
        if self._card is None:
            return None
        state = self._card.image_correction_state
        if category == "sharpen":
            return state.with_sharpen(key)  # type: ignore[arg-type]
        if category == "tone":
            return state.with_tone(key)  # type: ignore[arg-type]
        return None
