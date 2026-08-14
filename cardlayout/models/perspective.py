from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from PIL import Image

from cardlayout.models.detection import Point

PerspectiveConfidenceLevel = Literal["high", "medium", "low", "none"]
PerspectiveStatus = Literal["corrected", "review", "failed"]
PerspectiveMethod = Literal["automatic", "manual"]
Line = tuple[float, float, float]


@dataclass(slots=True)
class EdgeFitResult:
    """Robust fit and evidence quality for one physical card boundary."""

    name: str
    success: bool
    line: Line | None = None
    score: float = 0.0
    support_ratio: float = 0.0
    support_length_px: float = 0.0
    continuity: float = 0.0
    gradient_score: float = 0.0
    residual_px: float = 0.0
    orientation_delta_degrees: float = 0.0
    rough_offset_px: float = 0.0
    candidate_count: int = 0
    inlier_count: int = 0
    inferred: bool = False


@dataclass(slots=True)
class CornerRefinementResult:
    """Precise edge-fit geometry while preserving its rough input."""

    success: bool
    rough_corners: tuple[Point, ...]
    refined_corners: tuple[Point, ...]
    edge_results: tuple[EdgeFitResult, ...] = ()
    corner_confidences: tuple[float, ...] = ()
    confidence: float = 0.0
    roi_box: tuple[int, int, int, int] | None = None
    fallback_reason: str | None = None
    debug_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PerspectiveConfig:
    """Centralized quality, sizing, and validation policy for rectification."""

    preferred_width_px: int = 1200
    minimum_width_px: int = 320
    maximum_width_px: int = 1800
    max_upscale_factor: float = 1.15
    refinement_window_fraction: float = 0.018
    maximum_refinement_shift_fraction: float = 0.018
    boundary_tolerance_fraction: float = 0.025
    roi_padding_fraction: float = 0.10
    coarse_search_band_fraction: float = 0.065
    fine_search_band_fraction: float = 0.027
    edge_endpoint_trim_fraction: float = 0.045
    maximum_corner_displacement_fraction: float = 0.08
    maximum_edge_angle_delta_degrees: float = 13.0
    minimum_edge_fit_score: float = 0.43
    ransac_iterations: int = 48
    maximum_edge_candidates: int = 2200
    refinement_max_roi_long_edge: int = 2600
    minimum_area_px: float = 36.0
    minimum_edge_px: float = 6.0
    high_confidence: float = 0.74
    medium_confidence: float = 0.48

    def __post_init__(self) -> None:
        if not 64 <= self.minimum_width_px <= self.maximum_width_px:
            raise ValueError("Perspective output width limits are invalid")
        if not self.minimum_width_px <= self.preferred_width_px <= self.maximum_width_px:
            raise ValueError("Preferred width must be inside the output width limits")
        if self.max_upscale_factor < 1.0:
            raise ValueError("Maximum upscale factor cannot be less than 1")
        if not 0 < self.roi_padding_fraction < 0.5:
            raise ValueError("ROI padding fraction must be between 0 and 0.5")
        if not 0 < self.fine_search_band_fraction < self.coarse_search_band_fraction < 0.2:
            raise ValueError("Fine and coarse edge-search bands are invalid")
        if not 0 <= self.edge_endpoint_trim_fraction < 0.2:
            raise ValueError("Edge endpoint trim fraction is invalid")
        if not 0 < self.maximum_corner_displacement_fraction < 0.25:
            raise ValueError("Maximum corner displacement fraction is invalid")
        if self.ransac_iterations < 16 or self.maximum_edge_candidates < 200:
            raise ValueError("Robust edge-fitting limits are too small")
        if self.refinement_max_roi_long_edge < 512:
            raise ValueError("Refinement ROI resolution is too small")


@dataclass(slots=True)
class PerspectiveResult:
    """One non-destructive perspective-correction attempt."""

    success: bool
    source_points: tuple[Point, ...] = ()
    refined_points: tuple[Point, ...] = ()
    destination_points: tuple[Point, ...] = ()
    rectified_image: Image.Image | None = None
    confidence: float = 0.0
    confidence_level: PerspectiveConfidenceLevel = "none"
    status: PerspectiveStatus = "failed"
    method: PerspectiveMethod = "automatic"
    warning: str | None = None
    inferred_corner_count: int = 0
    output_dimensions: tuple[int, int] | None = None
    transform_matrix: tuple[tuple[float, float, float], ...] = ()
    corner_confidences: tuple[float, ...] = ()
    refinement_confidence: float = 0.0
    refinement_fallback_reason: str | None = None
    edge_results: tuple[EdgeFitResult, ...] = ()
    debug_info: dict[str, Any] = field(default_factory=dict)

    @property
    def status_text(self) -> str:
        if not self.success:
            return "Correction failed"
        if self.method == "manual":
            return "Manual correction applied"
        if self.status == "review":
            return "Detected · Review correction"
        return "Detected · Corrected"
