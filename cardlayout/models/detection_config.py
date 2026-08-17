from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DetectionScoreWeights:
    """Inspectable weights for the normalized candidate score."""

    area: float = 0.04
    geometry: float = 0.14
    aspect_ratio: float = 0.13
    edge_support: float = 0.14
    rectangularity: float = 0.07
    line_support: float = 0.11
    parallelism: float = 0.07
    perspective_quality: float = 0.05
    method_agreement: float = 0.05
    interior_detail: float = 0.04
    interior_complexity: float = 0.02
    border_contrast: float = 0.06
    foreground: float = 0.03
    nested_candidate: float = 0.05

    def __post_init__(self) -> None:
        total = sum(
            (
                self.area,
                self.geometry,
                self.aspect_ratio,
                self.edge_support,
                self.rectangularity,
                self.line_support,
                self.parallelism,
                self.perspective_quality,
                self.method_agreement,
                self.interior_detail,
                self.interior_complexity,
                self.border_contrast,
                self.foreground,
                self.nested_candidate,
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
    area_plateau_min_ratio: float = 0.012
    area_plateau_max_ratio: float = 0.26
    maximum_area_ratio: float = 0.90
    whole_frame_width_ratio: float = 0.975
    whole_frame_height_ratio: float = 0.975

    ratio_log_tolerance: float = 0.40
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
    border_touch_penalty: float = 0.060
    plain_rectangle_penalty: float = 0.08
    # Keep enough Stage-A alternatives for difficult perspective/occlusion
    # scenes. A real card can initially rank below crisp background rectangles.
    contextual_candidate_limit: int = 96
    contextual_analysis_long_edge: int = 1400
    oversize_start_ratio: float = 0.34
    oversize_full_ratio: float = 0.78
    maximum_oversize_penalty: float = 0.24
    nested_parent_penalty: float = 0.18
    nested_parent_min_area_ratio: float = 0.22
    nested_containment_ratio: float = 0.82
    nested_minimum_size_multiple: float = 1.20
    uniform_complexity_threshold: float = 0.25
    maximum_uniform_penalty: float = 0.15
    small_area_penalty_end_ratio: float = 0.020
    maximum_small_area_penalty: float = 0.14
    poor_ratio_score_threshold: float = 0.42
    maximum_ratio_penalty: float = 0.22
    appearance_border_band_fraction: float = 0.045
    one_edge_evidence_recovery: float = 0.05
    two_edge_evidence_recovery: float = 0.10
    ambiguity_close_gap: float = 0.045
    ambiguity_medium_gap: float = 0.09
    ambiguity_wide_gap: float = 0.15
    ambiguity_close_penalty: float = 0.20
    ambiguity_medium_penalty: float = 0.14
    ambiguity_wide_penalty: float = 0.06

    high_confidence: float = 0.74
    # Preserve a geometry-led best candidate for manual review in ambiguous
    # occlusion/perspective scenes instead of discarding useful full-card
    # corners just below the old 0.55 boundary.
    medium_confidence: float = 0.50
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
        if not (
            self.minimum_area_ratio
            < self.area_plateau_min_ratio
            < self.preferred_area_ratio
            < self.area_plateau_max_ratio
            < self.maximum_area_ratio
        ):
            raise ValueError("Candidate area prior is invalid")
        if not 0 < self.oversize_start_ratio < self.oversize_full_ratio < 1:
            raise ValueError("Oversize penalty ratios are invalid")
        if not 0 < self.nested_parent_min_area_ratio < self.oversize_full_ratio:
            raise ValueError("Nested parent area threshold is invalid")
        if not self.minimum_area_ratio < self.small_area_penalty_end_ratio < self.preferred_area_ratio:
            raise ValueError("Small-area penalty range is invalid")
        if not 0 < self.poor_ratio_score_threshold < 1 or not 0 <= self.maximum_ratio_penalty <= 0.30:
            raise ValueError("Aspect-ratio penalty parameters are invalid")
        if self.contextual_candidate_limit < 5 or self.contextual_analysis_long_edge < 400:
            raise ValueError("Contextual candidate ranking limits are invalid")
        if not 0 <= self.one_edge_evidence_recovery <= self.two_edge_evidence_recovery <= 0.15:
            raise ValueError("Partial-edge recovery bonuses are invalid")
        if not 0 < self.medium_confidence < self.high_confidence <= 1:
            raise ValueError("Confidence thresholds are invalid")
