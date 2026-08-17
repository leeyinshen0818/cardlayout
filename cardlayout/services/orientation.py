from __future__ import annotations

from PIL import Image

from cardlayout.models.orientation import (
    AutomaticOrientationResult,
    OrientationState,
)


def apply_orientation(source: Image.Image, state: OrientationState) -> Image.Image:
    """Render orientation once from unchanged geometry-corrected pixels."""
    if state.flip_horizontal and state.flip_vertical:
        return source.transpose(Image.Transpose.ROTATE_180)
    if state.flip_horizontal:
        return source.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if state.flip_vertical:
        return source.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return source.copy()


class OrientationAnalyzer:
    """Conservative local orientation stage that abstains without semantic OCR.

    Mirroring cannot be inferred reliably from geometry or edge density alone.
    CardLayout deliberately avoids destructive guesses until a dependable local
    text recognizer can be bundled without an external executable or cloud API.
    """

    def analyze(self, image: Image.Image) -> AutomaticOrientationResult:
        del image
        return AutomaticOrientationResult()
