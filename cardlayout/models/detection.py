from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from PIL import Image

ConfidenceLevel = Literal["high", "medium", "low", "none"]
Point = tuple[float, float]
BoundingBox = tuple[int, int, int, int]


@dataclass(slots=True)
class CardDetectionResult:
    """Detection output with geometry retained for future perspective work."""

    success: bool
    confidence: float
    confidence_level: ConfidenceLevel
    bounding_box: BoundingBox | None = None
    polygon_points: tuple[Point, ...] = ()
    cropped_image: Image.Image | None = None
    rotation_angle: float = 0.0
    method: str = "none"
    debug_info: dict[str, Any] = field(default_factory=dict)

    @property
    def status_text(self) -> str:
        percentage = int(round(self.confidence * 100))
        if self.success and self.method == "already_cropped":
            return "Already card-sized"
        if self.success and self.confidence_level == "high":
            return f"Detected · {percentage}%"
        if self.success:
            return f"Review recommended · {percentage}%"
        if self.confidence_level == "low":
            return f"Detection uncertain · {percentage}%"
        return "Detection failed"
