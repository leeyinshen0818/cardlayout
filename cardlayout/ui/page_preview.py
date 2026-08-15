from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap, QResizeEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QWidget

from cardlayout.models.card_side import CardSide, SideName
from cardlayout.services.layout_engine import LayoutEngine
from cardlayout.ui.image_utils import pil_to_qimage


class PagePreview(QWidget):
    position_adjust_requested = Signal(str, float)
    position_reset_requested = Signal(str)
    side_selected = Signal(str)
    selection_cleared = Signal()

    def __init__(self, engine: LayoutEngine, parent: object | None = None) -> None:
        super().__init__(parent)  # type: ignore[arg-type]
        self.engine = engine
        self.front: CardSide | None = None
        self.back: CardSide | None = None
        self.selected_side: SideName | None = None
        self._position_offsets: dict[SideName, float] = {
            "front": 0.0,
            "back": 0.0,
        }
        self.setMinimumSize(410, 560)
        self.setMouseTracking(True)
        self._build_position_controls()

    def _build_position_controls(self) -> None:
        self.position_controls = QFrame(self)
        self.position_controls.setObjectName("previewControls")
        self.position_controls.hide()

        self.selection_label = QLabel()
        self.selection_label.setObjectName("previewSelection")
        self.move_up_button = QPushButton("Move up 1 mm")
        self.move_up_button.setObjectName("previewControlButton")
        self.move_up_button.setAutoRepeat(True)
        self.move_up_button.clicked.connect(lambda: self._request_adjustment(-1.0))
        self.move_down_button = QPushButton("Move down 1 mm")
        self.move_down_button.setObjectName("previewControlButton")
        self.move_down_button.setAutoRepeat(True)
        self.move_down_button.clicked.connect(lambda: self._request_adjustment(1.0))
        self.reset_button = QPushButton("Reset")
        self.reset_button.setObjectName("previewControlButton")
        self.reset_button.clicked.connect(self._request_reset)
        close_button = QPushButton("Done")
        close_button.setObjectName("previewDoneButton")
        close_button.clicked.connect(self.clear_selection)

        controls_layout = QHBoxLayout(self.position_controls)
        controls_layout.setContentsMargins(12, 8, 8, 8)
        controls_layout.setSpacing(7)
        controls_layout.addWidget(self.selection_label, 1)
        controls_layout.addWidget(self.move_up_button)
        controls_layout.addWidget(self.move_down_button)
        controls_layout.addWidget(self.reset_button)
        controls_layout.addWidget(close_button)

    def set_position_offsets(self, front_mm: float, back_mm: float) -> None:
        self._position_offsets["front"] = front_mm
        self._position_offsets["back"] = back_mm
        self._update_selection_label()

    def set_sides(self, front: CardSide | None, back: CardSide | None) -> None:
        self.front = front
        self.back = back
        if self.selected_side is not None:
            selected = self._selected_card()
            if selected is None:
                self.clear_selection()
        self.update()

    def show_corrections(self, side: SideName) -> None:
        """Backward-compatible selection entry point for side-preview clicks."""
        self.select_side(side)

    def select_side(self, side: SideName, *, emit: bool = True) -> None:
        self.selected_side = side
        self._update_selection_label()
        self.position_controls.show()
        self.position_controls.raise_()
        self._position_controls()
        self.update()
        if emit:
            self.side_selected.emit(side)

    def paintEvent(self, event: object) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#e9edf3"))

        layout = self.engine.calculate()
        page_rect = self._page_rect()

        painter.fillRect(page_rect.translated(5, 7), QColor(30, 41, 59, 35))
        painter.fillRect(page_rect, Qt.GlobalColor.white)
        painter.setPen(QPen(QColor("#cbd3df"), 1))
        painter.drawRect(page_rect)

        self._draw_card(painter, page_rect, layout.front, self.front, "FRONT", "front")
        self._draw_card(painter, page_rect, layout.back, self.back, "BACK", "back")

    def _draw_card(self, painter, page_rect, rect, side, label, side_name) -> None:
        target = self._card_target(rect, page_rect)
        painter.save()
        painter.setClipRect(target)
        painter.fillRect(target, QColor("#f7f9fc"))
        if side is not None:
            pixmap = QPixmap.fromImage(pil_to_qimage(side.best_image))
            scaled = pixmap.scaled(
                target.size().toSize(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            image_rect = QRectF(
                target.center().x() - scaled.width() / 2,
                target.center().y() - scaled.height() / 2,
                scaled.width(),
                scaled.height(),
            )
            painter.drawPixmap(image_rect.toRect(), scaled)
        else:
            painter.setPen(QColor("#778396"))
            painter.drawText(target, Qt.AlignmentFlag.AlignCenter, f"{label}\nEmpty")
        painter.restore()
        if self.selected_side == side_name:
            painter.setPen(QPen(QColor("#245fc5"), 3))
        else:
            painter.setPen(QPen(QColor("#9aa7b8"), 1))
        painter.drawRect(target)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        side = self._side_at(event.position())
        if side is None:
            self.clear_selection()
        else:
            self.select_side(side)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        cursor = (
            Qt.CursorShape.PointingHandCursor
            if self._side_at(event.position()) is not None
            else Qt.CursorShape.ArrowCursor
        )
        self.setCursor(cursor)
        super().mouseMoveEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._position_controls()

    def clear_selection(self, *, emit: bool = True) -> None:
        self.selected_side = None
        self.position_controls.hide()
        self.update()
        if emit:
            self.selection_cleared.emit()

    def _selected_card(self) -> CardSide | None:
        if self.selected_side == "front":
            return self.front
        if self.selected_side == "back":
            return self.back
        return None

    def _request_adjustment(self, delta_mm: float) -> None:
        if self.selected_side is not None:
            self.position_adjust_requested.emit(self.selected_side, delta_mm)

    def _request_reset(self) -> None:
        if self.selected_side is not None:
            self.position_reset_requested.emit(self.selected_side)

    def _update_selection_label(self) -> None:
        if self.selected_side is None:
            return
        offset = self._position_offsets[self.selected_side]
        if abs(offset) < 0.0001:
            position = "Default position"
        else:
            direction = "down" if offset > 0 else "up"
            position = f"{abs(offset):g} mm {direction}"
        self.selection_label.setText(
            f"{self.selected_side.upper()}  ·  {position}"
        )

    def _position_controls(self) -> None:
        width = min(540, max(380, self.width() - 24))
        height = 54
        self.position_controls.setGeometry(
            (self.width() - width) // 2,
            self.height() - height - 14,
            width,
            height,
        )

    def _page_rect(self) -> QRectF:
        # Keep a modest breathing space while allowing the paper to use the
        # center workspace more effectively at normal desktop window sizes.
        margin = 18.0
        layout = self.engine.calculate()
        scale = min(
            max(1.0, self.width() - margin * 2) / layout.page_width_mm,
            max(1.0, self.height() - margin * 2) / layout.page_height_mm,
        )
        page_width = layout.page_width_mm * scale
        page_height = layout.page_height_mm * scale
        return QRectF(
            (self.width() - page_width) / 2,
            (self.height() - page_height) / 2,
            page_width,
            page_height,
        )

    def _card_target(self, rect, page_rect: QRectF) -> QRectF:
        layout = self.engine.calculate()
        x_scale = page_rect.width() / layout.page_width_mm
        y_scale = page_rect.height() / layout.page_height_mm
        return QRectF(
            page_rect.left() + rect.x * x_scale,
            page_rect.top() + rect.y * y_scale,
            rect.width * x_scale,
            rect.height * y_scale,
        )

    def _side_at(self, position) -> SideName | None:
        layout = self.engine.calculate()
        page_rect = self._page_rect()
        if self._card_target(layout.front, page_rect).contains(position):
            return "front"
        if self._card_target(layout.back, page_rect).contains(position):
            return "back"
        return None
