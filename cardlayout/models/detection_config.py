from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DetectionScoreWeights:
    """Inspectable weights for the normalized candidate score."""

    area: float = 0.06
    geometry: float = 0.15
    aspect_ratio: float = 0.23
    edge_support: float = 0.16
    rectangularity: float = 0.08
    line_support: float = 0.14
    method_agreement: float = 0.10
    interior_detail: float = 0.08

    def __post_init__(self) -> None:
        total = sum(
            (
                self.area,
                self.geometry,
                self.aspect_ratio,
                self.edge_support,
                self.rectangularity,
                self.line_support,
                self.method_agreement,
                self.interior_detail,
            )
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Detection score weights must total 1.0")


@dataclass(frozen=True, slots=True)
class CardDetectionConfig:
    """Central tuning parameters for the Phase 2.1 hybrid detector."""

    fast_long_edge: int = 1100
    detailed_long_edge: int = 1800
    fallback_long_edge: int = 2400
    early_accept_score: float = 0.77
    early_accept_margin: float = 0.10

    minimum_area_ratio: float = 0.0012
    tiny_area_ratio: float = 0.008
    preferred_area_ratio: float = 0.10
    maximum_area_ratio: float = 0.90
    whole_frame_width_ratio: float = 0.975
    whole_frame_height_ratio: float = 0.975

    ratio_log_tolerance: float = 0.46
    contour_min_dimension_px: int = 14
    contour_epsilon_ratio: float = 0.022
    edge_band_fraction: float = 0.005

    hough_threshold: int = 32
    line_min_length_fraction: float = 0.055
    line_max_gap_fraction: float = 0.025
    line_angle_tolerance_degrees: float = 11.0
    line_cluster_distance_fraction: float = 0.012
    line_min_separation_fraction: float = 0.045
    maximum_hough_lines: int = 80
    maximum_line_clusters: int = 8

    one_inferred_edge_penalty: float = 0.10
    two_inferred_edges_penalty: float = 0.18
    border_touch_penalty: float = 0.035
    plain_rectangle_penalty: float = 0.08
    ambiguity_close_gap: float = 0.045
    ambiguity_medium_gap: float = 0.09
    ambiguity_wide_gap: float = 0.15
    ambiguity_close_penalty: float = 0.20
    ambiguity_medium_penalty: float = 0.14
    ambiguity_wide_penalty: float = 0.06

    high_confidence: float = 0.74
    medium_confidence: float = 0.55
    low_confidence: float = 0.36
    crop_safety_margin_fraction: float = 0.015
    weights: DetectionScoreWeights = field(default_factory=DetectionScoreWeights)

    def __post_init__(self) -> None:
        if not (
            0 < self.fast_long_edge
            <= self.detailed_long_edge
            <= self.fallback_long_edge
        ):
            raise ValueError("Detection working scales must be positive and increasing")
        if not 0 < self.minimum_area_ratio < self.maximum_area_ratio < 1:
            raise ValueError("Candidate area limits are invalid")
        if not 0 < self.medium_confidence < self.high_confidence <= 1:
            raise ValueError("Confidence thresholds are invalid")
