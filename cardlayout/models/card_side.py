from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from cardlayout.models.detection import CardDetectionResult
from cardlayout.models.image_correction import ImageCorrectionState
from cardlayout.models.perspective import PerspectiveResult
from cardlayout.services.image_corrections import apply_image_correction

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
    image_correction_state: ImageCorrectionState = field(
        default_factory=ImageCorrectionState
    )
    corrected_image: Image.Image | None = None
    detector_input_image: Image.Image | None = None
    original_pdf_render: Image.Image | None = None
    normalized_pdf_raster: Image.Image | None = None
    source_diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def processing_raster(self) -> Image.Image:
        """Full-quality canonical raster shared by detection and perspective."""
        return self.detector_input_image or self.original_image

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
        else:
            self.detected_image = None
        self._refresh_image_correction()
        self.processed_image = self.best_image

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
        if not self.image_correction_state.is_normal and self.corrected_image is not None:
            return self.corrected_image
        return self.geometry_image

    @property
    def geometry_image(self) -> Image.Image:
        """Best geometry-only image before Phase 4 appearance corrections."""
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

    def apply_image_correction(self, state: ImageCorrectionState) -> None:
        self.image_correction_state = state
        self._refresh_image_correction()
        self.processed_image = self.best_image

    def reset_correction(self) -> None:
        """Discard manual geometry and restore the current automatic result."""
        self.manual_perspective_result = None
        self._refresh_processed_stage()

    def reset_user_edits(self) -> None:
        """Restore automatic geometry and Normal appearance without reloading."""
        self.manual_perspective_result = None
        self.image_correction_state = ImageCorrectionState()
        self.corrected_image = None
        self._refresh_processed_stage()

    def _refresh_processed_stage(self) -> None:
        active = self.active_perspective_result
        if active is not None and active.success and active.rectified_image is not None:
            self.rectified_image = active.rectified_image
        else:
            self.rectified_image = None
        self._refresh_image_correction()
        self.processed_image = self.best_image

    def _refresh_image_correction(self) -> None:
        if self.image_correction_state.is_normal:
            self.corrected_image = None
        else:
            self.corrected_image = apply_image_correction(
                self.geometry_image, self.image_correction_state
            )

    def reset_detection(self) -> None:
        self.detected_image = None
        self.detection_result = None
        self.rectified_image = None
        self.automatic_perspective_result = None
        self.manual_perspective_result = None
        self.image_correction_state = ImageCorrectionState()
        self.corrected_image = None
        self.processed_image = self.original_image.copy()
