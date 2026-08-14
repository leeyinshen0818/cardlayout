from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from PIL import Image

from cardlayout.models.detection import CardDetectionResult

SideName = Literal["front", "back"]
SourceType = Literal["image", "pdf"]


@dataclass(slots=True)
class CardSide:
    """One normalized card face, independent of its original file format."""

    side: SideName
    source_path: Path
    source_type: SourceType
    source_page: int | None
    original_image: Image.Image
    processed_image: Image.Image
    detected_image: Image.Image | None = None
    detection_result: CardDetectionResult | None = None

    @property
    def display_name(self) -> str:
        if self.source_type == "pdf" and self.source_page is not None:
            return f"{self.source_path.name} — page {self.source_page}"
        return self.source_path.name

    def assigned_to(self, side: SideName) -> CardSide:
        """Return a new assignment without reloading or copying image pixels."""
        return replace(self, side=side)

    def apply_detection(self, result: CardDetectionResult) -> None:
        """Apply a safe detection result without altering the normalized source."""
        self.detection_result = result
        if result.success and result.cropped_image is not None:
            self.detected_image = result.cropped_image
            self.processed_image = self.detected_image
        else:
            self.detected_image = None
            self.processed_image = self.original_image.copy()

    def reset_detection(self) -> None:
        self.detected_image = None
        self.detection_result = None
        self.processed_image = self.original_image.copy()
