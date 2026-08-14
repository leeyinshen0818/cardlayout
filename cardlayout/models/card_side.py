from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from PIL import Image

from cardlayout.models.detection import CardDetectionResult
from cardlayout.models.perspective import PerspectiveResult

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
    rectified_image: Image.Image | None = None
    automatic_perspective_result: PerspectiveResult | None = None
    manual_perspective_result: PerspectiveResult | None = None

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
        self.automatic_perspective_result = None
        self.manual_perspective_result = None
        self.rectified_image = None
        self.detection_result = result
        if result.success and result.cropped_image is not None:
            self.detected_image = result.cropped_image
            self.processed_image = self.detected_image
        else:
            self.detected_image = None
            self.processed_image = self.original_image.copy()

    @property
    def active_perspective_result(self) -> PerspectiveResult | None:
        if self.manual_perspective_result is not None:
            return self.manual_perspective_result
        return self.automatic_perspective_result

    @property
    def has_manual_correction(self) -> bool:
        return self.manual_perspective_result is not None

    @property
    def best_image(self) -> Image.Image:
        """Image used by both the A4 preview and every exporter."""
        if self.manual_perspective_result is not None:
            manual = self.manual_perspective_result.rectified_image
            if self.manual_perspective_result.success and manual is not None:
                return manual
        if self.automatic_perspective_result is not None:
            automatic = self.automatic_perspective_result.rectified_image
            if self.automatic_perspective_result.success and automatic is not None:
                return automatic
        if self.detected_image is not None:
            return self.detected_image
        return self.original_image

    @property
    def correction_status_text(self) -> str:
        active = self.active_perspective_result
        if active is not None:
            return active.status_text
        if self.detection_result is not None:
            return self.detection_result.status_text
        return "Not processed"

    def apply_automatic_correction(self, result: PerspectiveResult) -> None:
        self.automatic_perspective_result = result
        self.manual_perspective_result = None
        self._refresh_processed_stage()

    def apply_manual_correction(self, result: PerspectiveResult) -> None:
        if not result.success or result.rectified_image is None:
            raise ValueError("A successful perspective result is required")
        self.manual_perspective_result = result
        self._refresh_processed_stage()

    def reset_correction(self) -> None:
        """Discard manual geometry and restore the current automatic result."""
        self.manual_perspective_result = None
        self._refresh_processed_stage()

    def _refresh_processed_stage(self) -> None:
        active = self.active_perspective_result
        if active is not None and active.success and active.rectified_image is not None:
            self.rectified_image = active.rectified_image
        else:
            self.rectified_image = None
        self.processed_image = self.best_image

    def reset_detection(self) -> None:
        self.detected_image = None
        self.detection_result = None
        self.rectified_image = None
        self.automatic_perspective_result = None
        self.manual_perspective_result = None
        self.processed_image = self.original_image.copy()
