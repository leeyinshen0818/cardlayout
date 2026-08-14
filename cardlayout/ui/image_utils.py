from __future__ import annotations

from PIL import Image
from PySide6.QtGui import QImage


def pil_to_qimage(image: Image.Image) -> QImage:
    rgba = image.convert("RGBA")
    qimage = QImage(
        rgba.tobytes("raw", "RGBA"),
        rgba.width,
        rgba.height,
        rgba.width * 4,
        QImage.Format.Format_RGBA8888,
    )
    return qimage.copy()

