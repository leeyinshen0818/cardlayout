from __future__ import annotations

from collections.abc import Callable

from PIL import Image
from PySide6.QtCore import QPointF, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QBrush, QPen, QPixmap, QPolygonF, QTransform, QWheelEvent
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from cardlayout.models.detection import Point
from cardlayout.models.perspective import PerspectiveResult
from cardlayout.services.perspective_corrector import PerspectiveCorrector
from cardlayout.ui.image_utils import pil_to_qimage


class _CornerHandle(QGraphicsEllipseItem):
    def __init__(
        self,
        point: Point,
        bounds: tuple[int, int],
        moved: Callable[[], None],
    ) -> None:
        super().__init__(-8, -8, 16, 16)
        self._bounds = bounds
        self._moved = moved
        self.setBrush(QBrush(QColor("#ffffff")))
        self.setPen(QPen(QColor("#245fc5"), 3))
        self.setZValue(5)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setPos(*point)

    def itemChange(self, change, value):  # type: ignore[no-untyped-def]
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            point = value
            return QPointF(
                min(max(0.0, point.x()), self._bounds[0] - 1.0),
                min(max(0.0, point.y()), self._bounds[1] - 1.0),
            )
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._moved()
        return result


class CornerEditorCanvas(QGraphicsView):
    """Image-coordinate corner canvas with stable zoom and pan mapping."""

    corners_changed = Signal(object)
    CORNER_NAMES = ("TL", "TR", "BR", "BL")

    def __init__(
        self,
        image: Image.Image,
        points: tuple[Point, ...],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._image_size = image.size
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setPixmap(QPixmap.fromImage(pil_to_qimage(image)))
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(0, 0, image.width, image.height)
        self._polygon = QGraphicsPolygonItem()
        self._polygon.setPen(QPen(QColor("#2f7cf6"), 3))
        self._polygon.setBrush(QBrush(QColor(47, 124, 246, 28)))
        self._polygon.setZValue(3)
        self._scene.addItem(self._polygon)
        self._handles: list[_CornerHandle] = []
        self._labels: list[QGraphicsSimpleTextItem] = []
        for name, point in zip(self.CORNER_NAMES, points, strict=True):
            handle = _CornerHandle(point, image.size, self._handles_moved)
            self._scene.addItem(handle)
            label = QGraphicsSimpleTextItem(name)
            label.setBrush(QBrush(QColor("#ffffff")))
            label.setPen(QPen(QColor("#153d7a"), 2))
            label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            label.setZValue(6)
            self._scene.addItem(label)
            self._handles.append(handle)
            self._labels.append(label)
        self.setRenderHints(self.renderHints())
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QBrush(QColor("#172033")))
        self.setMinimumSize(520, 420)
        self._handles_moved()
        QTimer.singleShot(0, self.fit_image)

    @property
    def corners(self) -> tuple[Point, ...]:
        return tuple((handle.pos().x(), handle.pos().y()) for handle in self._handles)

    def set_corners(self, points: tuple[Point, ...]) -> None:
        for handle, point in zip(self._handles, points, strict=True):
            handle.setPos(*point)
        self._handles_moved()

    def fit_image(self) -> None:
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_actual_size(self) -> None:
        self.setTransform(QTransform())

    def wheelEvent(self, event: QWheelEvent) -> None:
        old_scale = self.transform().m11()
        factor = 1.22 if event.angleDelta().y() > 0 else 1 / 1.22
        new_scale = old_scale * factor
        if 0.03 <= new_scale <= 16.0:
            self.scale(factor, factor)
        event.accept()

    def _handles_moved(self) -> None:
        if len(self._handles) != 4:
            return
        polygon = QPolygonF([handle.pos() for handle in self._handles])
        self._polygon.setPolygon(polygon)
        for handle, label in zip(self._handles, self._labels, strict=True):
            label.setPos(handle.pos() + QPointF(10, -24))
        self.corners_changed.emit(self.corners)


class CornerEditorDialog(QDialog):
    """Focused four-corner workflow with debounced rectified preview."""

    def __init__(
        self,
        image: Image.Image,
        automatic_points: tuple[Point, ...],
        corrector: PerspectiveCorrector,
        detector_confidence: float,
        current_points: tuple[Point, ...] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._image = image
        self._corrector = corrector
        self._detector_confidence = detector_confidence
        self._automatic_points = automatic_points
        self._result: PerspectiveResult | None = None
        self.setWindowTitle("Adjust Card Corners")
        self.resize(1120, 720)
        self.setMinimumSize(820, 560)

        self.canvas = CornerEditorCanvas(image, current_points or automatic_points)
        self.preview = QLabel("Rectified preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumWidth(300)
        self.preview.setStyleSheet(
            "background:#eef2f7; border:1px solid #cbd5e1; color:#64748b;"
        )
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.preview)
        splitter.setSizes([750, 350])

        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color:#b42318; font-weight:600;")
        self.fit_button = QPushButton("Fit")
        self.fit_button.clicked.connect(self.canvas.fit_image)
        self.actual_button = QPushButton("100%")
        self.actual_button.clicked.connect(self.canvas.set_actual_size)
        self.reset_button = QPushButton("Reset to Automatic")
        self.reset_button.clicked.connect(
            lambda: self.canvas.set_corners(self._automatic_points)
        )
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Drag TL, TR, BR, and BL onto the physical card corners."))
        controls.addStretch()
        controls.addWidget(self.fit_button)
        controls.addWidget(self.actual_button)
        controls.addWidget(self.reset_button)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.apply_button = self.buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.buttons.clicked.connect(self._button_clicked)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(90)
        self._preview_timer.timeout.connect(self._update_rectified_preview)
        self.canvas.corners_changed.connect(self._corners_changed)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.warning)
        layout.addWidget(self.buttons)
        self._corners_changed(self.canvas.corners)

    @property
    def result(self) -> PerspectiveResult | None:
        return self._result

    def _corners_changed(self, points: object) -> None:
        corners = points  # Signal payload is a tuple of image-coordinate points.
        valid, warning = self._corrector.validate_quad(
            corners, self._image.size  # type: ignore[arg-type]
        )
        self.apply_button.setEnabled(valid)
        self.warning.setText("" if valid else (warning or "Invalid corners"))
        if valid:
            self._preview_timer.start()
        else:
            self._preview_timer.stop()
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Move the handles to form a valid card outline")

    def _update_rectified_preview(self) -> None:
        result = self._make_result()
        if not result.success or result.rectified_image is None:
            self.apply_button.setEnabled(False)
            self.warning.setText(result.warning or "Correction failed")
            return
        pixmap = QPixmap.fromImage(pil_to_qimage(result.rectified_image))
        target = self.preview.size()
        target.setWidth(max(1, target.width() - 20))
        target.setHeight(max(1, target.height() - 20))
        self.preview.setPixmap(
            pixmap.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.preview.setText("")

    def _make_result(self) -> PerspectiveResult:
        return self._corrector.correct(
            self._image,
            self.canvas.corners,
            detector_confidence=self._detector_confidence,
            method="manual",
            refine=False,
        )

    def _button_clicked(self, button: QPushButton) -> None:
        role = self.buttons.buttonRole(button)
        if role == QDialogButtonBox.ButtonRole.ApplyRole:
            result = self._make_result()
            if result.success:
                self._result = result
                self.accept()
            else:
                self.warning.setText(result.warning or "Correction failed")
        elif role == QDialogButtonBox.ButtonRole.RejectRole:
            self.reject()
