from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from math import atan2, cos, degrees, exp, log, pi, sin
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image

from cardlayout.models.card_size import CardSizePreset
from cardlayout.models.detection import CardDetectionResult
from cardlayout.models.detection_config import CardDetectionConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _DetectedLine:
    p1: np.ndarray
    p2: np.ndarray
    angle: float
    length: float


@dataclass(slots=True)
class _LineCluster:
    offset: float
    coverage: float
    projection_min: float
    projection_max: float
    segment_count: int
    intervals: list[tuple[float, float]]


@dataclass(slots=True)
class _Candidate:
    polygon: np.ndarray
    source_methods: set[str]
    scales_used: set[int]
    total_score: float
    geometry_score: float
    ratio_score: float
    edge_score: float
    area_score: float
    rectangularity_score: float
    line_support_score: float
    agreement_score: float
    interior_detail_score: float
    interior_complexity_score: float
    border_contrast_score: float
    foreground_score: float
    nested_candidate_score: float
    background_penalty: float
    oversize_penalty: float
    uniform_surface_penalty: float
    small_area_penalty: float
    ratio_penalty: float
    partial_evidence_bonus: float
    occlusion_penalty: float
    inferred_edges: int
    area_ratio: float
    observed_ratio: float
    border_touches: int
    full_card_boundary_score: float = 0.0
    candidate_coverage_score: float = 0.0
    internal_subregion_penalty: float = 0.0
    candidate_id: str = ""
    contains_candidates: tuple[str, ...] = ()
    contained_by_candidates: tuple[str, ...] = ()
    overlap_candidates: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    contextual_scored: bool = False

    @property
    def bounding_box(self) -> tuple[int, int, int, int]:
        return (
            int(np.floor(self.polygon[:, 0].min())),
            int(np.floor(self.polygon[:, 1].min())),
            int(np.ceil(self.polygon[:, 0].max())),
            int(np.ceil(self.polygon[:, 1].max())),
        )

    @property
    def strategy_groups(self) -> set[str]:
        return {method.split(":", 1)[0] for method in self.source_methods}

    def debug_values(self) -> dict[str, Any]:
        values = {
            "source_methods": sorted(self.source_methods),
            "scales_used": sorted(self.scales_used),
            "total_score": round(self.total_score, 4),
            "geometry_score": round(self.geometry_score, 4),
            "ratio_score": round(self.ratio_score, 4),
            "edge_score": round(self.edge_score, 4),
            "area_score": round(self.area_score, 4),
            "rectangularity_score": round(self.rectangularity_score, 4),
            "line_support_score": round(self.line_support_score, 4),
            "agreement_score": round(self.agreement_score, 4),
            "interior_detail_score": round(self.interior_detail_score, 4),
            "interior_complexity_score": round(self.interior_complexity_score, 4),
            "border_contrast_score": round(self.border_contrast_score, 4),
            "foreground_score": round(self.foreground_score, 4),
            "nested_candidate_score": round(self.nested_candidate_score, 4),
            "full_card_boundary_score": round(self.full_card_boundary_score, 4),
            "candidate_coverage_score": round(self.candidate_coverage_score, 4),
            "internal_subregion_penalty": round(self.internal_subregion_penalty, 4),
            "background_penalty": round(self.background_penalty, 4),
            "oversize_penalty": round(self.oversize_penalty, 4),
            "uniform_surface_penalty": round(self.uniform_surface_penalty, 4),
            "small_area_penalty": round(self.small_area_penalty, 4),
            "ratio_penalty": round(self.ratio_penalty, 4),
            "partial_evidence_bonus": round(self.partial_evidence_bonus, 4),
            "occlusion_penalty": round(self.occlusion_penalty, 4),
            "inferred_edges": self.inferred_edges,
            "area_ratio": round(self.area_ratio, 5),
            "observed_ratio": round(self.observed_ratio, 4),
            "border_touches": self.border_touches,
            "bounding_box": self.bounding_box,
            "candidate_id": self.candidate_id,
            "contains_candidates": list(self.contains_candidates),
            "contained_by_candidates": list(self.contained_by_candidates),
            "overlap_candidates": list(self.overlap_candidates),
            "rejection_reasons": list(self.rejection_reasons),
            "contextual_scored": self.contextual_scored,
        }
        # Phase 2 compatibility: geometry_score supersedes shape_score.
        values["shape_score"] = values["geometry_score"]
        return values


@dataclass(slots=True)
class _PassResult:
    name: str
    scale_used: int
    candidates: list[_Candidate]
    line_segments: list[_DetectedLine]
    working_rgb: np.ndarray
    edge_map: np.ndarray
    stage_images: dict[str, np.ndarray] = field(default_factory=dict)


class CardDetector:
    """Staged multi-scale hybrid detector for physical cards in messy photos."""

    def __init__(
        self,
        card_size: CardSizePreset,
        max_working_dimension: int | None = None,
        debug: bool = False,
        config: CardDetectionConfig | None = None,
    ) -> None:
        self.card_size = card_size
        self.config = config or CardDetectionConfig()
        if max_working_dimension is not None:
            # Backward-compatible Phase 2 override: use the cap for every pass.
            cap = max(400, max_working_dimension)
            self.config = CardDetectionConfig(
                fast_long_edge=min(self.config.fast_long_edge, cap),
                detailed_long_edge=cap,
                fallback_long_edge=cap,
                weights=self.config.weights,
            )
        self.debug = debug

    def detect(self, image: Image.Image) -> CardDetectionResult:
        started = perf_counter()
        original = np.asarray(image.convert("RGB"))
        original_height, original_width = original.shape[:2]
        debug_info: dict[str, Any] = {
            "original_size": (original_width, original_height),
            "passes": [],
        }
        if original_width < 40 or original_height < 40:
            return self._finalize(
                self._failure("Image is too small for card detection"),
                started,
                debug_info,
            )

        all_candidates: list[_Candidate] = []
        pass_results: list[_PassResult] = []

        fast = self._run_pass(
            original,
            target_long_edge=self.config.fast_long_edge,
            name="fast_contour",
            extended_color=False,
            include_lines=False,
            include_partial=False,
        )
        pass_results.append(fast)
        all_candidates.extend(fast.candidates)
        fused = self._contextual_rank(self._fuse_candidates(all_candidates), original)

        if self._looks_already_cropped(
            fast, fused, original_width, original_height
        ):
            result = CardDetectionResult(
                success=True,
                confidence=0.86,
                confidence_level="high",
                bounding_box=(0, 0, original_width, original_height),
                polygon_points=(
                    (0.0, 0.0),
                    (float(original_width - 1), 0.0),
                    (float(original_width - 1), float(original_height - 1)),
                    (0.0, float(original_height - 1)),
                ),
                cropped_image=image.convert("RGB").copy(),
                rotation_angle=0.0,
                method="already_cropped",
            )
            debug_info["reason"] = "Input already appears card-sized"
            self._attach_pass_debug(debug_info, pass_results, fused, original)
            return self._finalize(result, started, debug_info)

        if self.debug or not self._can_early_accept(fused):
            detailed = self._run_pass(
                original,
                target_long_edge=self.config.detailed_long_edge,
                name="detailed_hybrid",
                extended_color=True,
                include_lines=True,
                include_partial=False,
            )
            pass_results.append(detailed)
            all_candidates.extend(detailed.candidates)
            fused = self._contextual_rank(self._fuse_candidates(all_candidates), original)

        if self._needs_reconstruction(fused):
            reconstruction = self._run_pass(
                original,
                target_long_edge=self.config.fallback_long_edge,
                name="occlusion_reconstruction",
                extended_color=True,
                include_lines=True,
                include_partial=True,
            )
            pass_results.append(reconstruction)
            all_candidates.extend(reconstruction.candidates)
            fused = self._contextual_rank(self._fuse_candidates(all_candidates), original)

        self._attach_pass_debug(debug_info, pass_results, fused, original)
        if not fused:
            return self._finalize(
                self._failure("No plausible card region was found"),
                started,
                debug_info,
            )

        best = fused[0]
        second = fused[1] if len(fused) > 1 else None
        confidence, reasoning = self._confidence(best, second)
        debug_info["candidate_count"] = len(fused)
        debug_info["candidates"] = [candidate.debug_values() for candidate in fused[:25]]
        debug_info["selected_candidate"] = best.debug_values()
        debug_info["inferred_edge_count"] = best.inferred_edges
        debug_info["best_score"] = round(best.total_score, 4)
        debug_info["second_best_score"] = (
            round(second.total_score, 4) if second is not None else None
        )
        debug_info["confidence_reasoning"] = reasoning
        debug_info["ambiguity_penalty"] = reasoning["ambiguity_penalty"]

        polygon = np.asarray(best.polygon, dtype=np.float32)
        bounding_box = self._axis_aligned_bounds(
            polygon, original_width, original_height
        )
        rotation_angle = self._long_edge_angle(polygon)
        method = "hybrid:" + "+".join(sorted(best.strategy_groups))

        if confidence < self.config.medium_confidence:
            level = (
                "low" if confidence >= self.config.low_confidence else "none"
            )
            return self._finalize(
                CardDetectionResult(
                    success=False,
                    confidence=confidence,
                    confidence_level=level,
                    bounding_box=bounding_box,
                    polygon_points=self._polygon_tuple(polygon),
                    rotation_angle=rotation_angle,
                    method=method,
                ),
                started,
                debug_info,
            )

        cropped, applied_angle = self._rotated_crop(original, polygon)
        if cropped.width < 20 or cropped.height < 20:
            return self._finalize(
                CardDetectionResult(
                    success=False,
                    confidence=min(confidence, self.config.low_confidence),
                    confidence_level="low",
                    bounding_box=bounding_box,
                    polygon_points=self._polygon_tuple(polygon),
                    rotation_angle=rotation_angle,
                    method=method,
                ),
                started,
                debug_info,
            )

        return self._finalize(
            CardDetectionResult(
                success=True,
                confidence=confidence,
                confidence_level=(
                    "high"
                    if confidence >= self.config.high_confidence
                    else "medium"
                ),
                bounding_box=bounding_box,
                polygon_points=self._polygon_tuple(polygon),
                cropped_image=cropped,
                rotation_angle=applied_angle,
                method=method,
            ),
            started,
            debug_info,
        )

    def _run_pass(
        self,
        original: np.ndarray,
        target_long_edge: int,
        name: str,
        extended_color: bool,
        include_lines: bool,
        include_partial: bool,
    ) -> _PassResult:
        working, scale_x, scale_y = self._working_copy(original, target_long_edge)
        stages, edge_map = self._preprocess(working, extended_color)
        scale_used = max(working.shape[:2])
        candidates = self._contour_candidates(
            stages,
            edge_map,
            working.shape[:2],
            scale_x,
            scale_y,
            scale_used,
        )
        lines: list[_DetectedLine] = []
        if include_lines:
            lines = self._detect_lines(edge_map)
            candidates.extend(
                self._line_candidates(
                    lines,
                    edge_map,
                    working.shape[:2],
                    scale_x,
                    scale_y,
                    scale_used,
                    include_partial,
                )
            )
        candidates = sorted(
            candidates, key=lambda candidate: candidate.total_score, reverse=True
        )[:500]
        return _PassResult(
            name=name,
            scale_used=scale_used,
            candidates=candidates,
            line_segments=lines,
            working_rgb=working,
            edge_map=edge_map,
            stage_images=stages,
        )

    @staticmethod
    def _working_copy(
        original: np.ndarray, target_long_edge: int
    ) -> tuple[np.ndarray, float, float]:
        height, width = original.shape[:2]
        largest = max(width, height)
        if largest <= target_long_edge:
            return original.copy(), 1.0, 1.0
        factor = target_long_edge / largest
        working_width = max(1, int(round(width * factor)))
        working_height = max(1, int(round(height * factor)))
        working = cv2.resize(
            original,
            (working_width, working_height),
            interpolation=cv2.INTER_AREA,
        )
        return working, width / working_width, height / working_height

    def _preprocess(
        self, rgb: np.ndarray, extended_color: bool
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        luminance = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0]
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        gray_edges = self._auto_canny(blurred)
        kernel = self._morphology_kernel(gray.shape)
        linked_edges = cv2.morphologyEx(
            gray_edges, cv2.MORPH_CLOSE, kernel, iterations=2
        )
        linked_edges = cv2.dilate(
            linked_edges, np.ones((3, 3), np.uint8), iterations=1
        )
        _, otsu = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        stages: dict[str, np.ndarray] = {
            "grayscale": gray,
            "edges": linked_edges,
            "otsu": cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1),
        }
        stages["otsu_inverse"] = cv2.bitwise_not(stages["otsu"])
        combined_edges = linked_edges.copy()

        if extended_color:
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            color_edges = np.zeros_like(gray)
            for channel in (luminance, lab[:, :, 1], lab[:, :, 2], hsv[:, :, 1]):
                channel_edges = self._auto_canny(cv2.GaussianBlur(channel, (5, 5), 0))
                color_edges = cv2.bitwise_or(color_edges, channel_edges)
            color_edges = cv2.morphologyEx(
                color_edges, cv2.MORPH_CLOSE, kernel, iterations=1
            )
            combined_edges = cv2.bitwise_or(combined_edges, color_edges)
            block_size = max(15, int(round(min(gray.shape) * 0.045)))
            if block_size % 2 == 0:
                block_size += 1
            block_size = min(block_size, 61)
            adaptive = cv2.adaptiveThreshold(
                cv2.GaussianBlur(luminance, (5, 5), 0),
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                5,
            )
            stages.update(
                {
                    "color_edges": color_edges,
                    "adaptive": cv2.morphologyEx(
                        adaptive, cv2.MORPH_CLOSE, kernel, iterations=1
                    ),
                }
            )
            stages["adaptive_inverse"] = cv2.bitwise_not(stages["adaptive"])
            stages["combined_edges"] = combined_edges
            for channel_name, channel in (
                ("lab_a", lab[:, :, 1]),
                ("lab_b", lab[:, :, 2]),
                ("saturation", hsv[:, :, 1]),
            ):
                _, color_mask = cv2.threshold(
                    cv2.GaussianBlur(channel, (5, 5), 0),
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )
                color_mask = cv2.morphologyEx(
                    color_mask, cv2.MORPH_CLOSE, kernel, iterations=2
                )
                stages[f"{channel_name}_otsu"] = color_mask
                stages[f"{channel_name}_otsu_inverse"] = cv2.bitwise_not(
                    color_mask
                )
        return stages, combined_edges

    @staticmethod
    def _auto_canny(channel: np.ndarray) -> np.ndarray:
        median = float(np.median(channel))
        lower = int(max(12, 0.58 * median))
        upper = int(min(255, max(lower + 25, 1.42 * median)))
        return cv2.Canny(channel, lower, upper)

    @staticmethod
    def _morphology_kernel(shape: tuple[int, int]) -> np.ndarray:
        size = max(3, int(round(min(shape) * 0.005)))
        if size % 2 == 0:
            size += 1
        return cv2.getStructuringElement(cv2.MORPH_RECT, (min(size, 9), min(size, 9)))

    def _contour_candidates(
        self,
        stages: dict[str, np.ndarray],
        edge_map: np.ndarray,
        image_shape: tuple[int, int],
        scale_x: float,
        scale_y: float,
        scale_used: int,
    ) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for stage_name, stage in stages.items():
            if stage_name in {"grayscale", "color_edges"}:
                continue
            contours, _ = cv2.findContours(
                stage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = sorted(
                contours, key=lambda contour: abs(cv2.contourArea(contour)), reverse=True
            )[:500]
            for contour in contours:
                area = abs(float(cv2.contourArea(contour)))
                if area <= 0:
                    continue
                perimeter = cv2.arcLength(contour, True)
                if perimeter <= 0:
                    continue
                rotated = cv2.minAreaRect(contour)
                rect_width, rect_height = rotated[1]
                if (
                    rect_width < self.config.contour_min_dimension_px
                    or rect_height < self.config.contour_min_dimension_px
                ):
                    continue
                rotated_area = float(rect_width * rect_height)
                if rotated_area <= 0:
                    continue
                rectangularity = float(np.clip(area / rotated_area, 0.0, 1.0))
                approximation = cv2.approxPolyDP(
                    contour, self.config.contour_epsilon_ratio * perimeter, True
                )
                vertices = len(approximation)
                angle_score = (
                    self._angle_score(approximation) if vertices == 4 else 0.0
                )
                convex = vertices == 4 and cv2.isContourConvex(approximation)
                vertex_score = {4: 1.0, 5: 0.76, 6: 0.56}.get(vertices, 0.28)
                geometry = float(
                    np.clip(
                        vertex_score * 0.66
                        + (0.14 if convex else 0.0)
                        + angle_score * 0.20,
                        0.0,
                        1.0,
                    )
                )
                box = cv2.boxPoints(rotated).astype(np.float32)
                rotated_candidate = self._make_candidate(
                    box,
                    source=f"rotated_rect:{stage_name}",
                    scale_used=scale_used,
                    edge_map=edge_map,
                    image_shape=image_shape,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    geometry_score=max(0.58, geometry),
                    rectangularity_score=rectangularity,
                    line_support_score=0.0,
                    inferred_edges=0,
                )
                if rotated_candidate is not None:
                    candidates.append(rotated_candidate)
                if vertices == 4 and convex:
                    contour_candidate = self._make_candidate(
                        approximation.reshape(4, 2).astype(np.float32),
                        source=f"contour:{stage_name}",
                        scale_used=scale_used,
                        edge_map=edge_map,
                        image_shape=image_shape,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        geometry_score=geometry,
                        rectangularity_score=rectangularity,
                        line_support_score=0.0,
                        inferred_edges=0,
                    )
                    if contour_candidate is not None:
                        candidates.append(contour_candidate)
        return candidates

    def _detect_lines(self, edges: np.ndarray) -> list[_DetectedLine]:
        minimum_dimension = min(edges.shape)
        raw = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=self.config.hough_threshold,
            minLineLength=max(
                18, int(round(minimum_dimension * self.config.line_min_length_fraction))
            ),
            maxLineGap=max(
                6, int(round(minimum_dimension * self.config.line_max_gap_fraction))
            ),
        )
        if raw is None:
            return []
        detected: list[_DetectedLine] = []
        for values in raw.reshape(-1, 4):
            p1 = np.asarray(values[:2], dtype=np.float32)
            p2 = np.asarray(values[2:], dtype=np.float32)
            vector = p2 - p1
            length = float(np.linalg.norm(vector))
            if length <= 0:
                continue
            angle = atan2(float(vector[1]), float(vector[0])) % pi
            detected.append(_DetectedLine(p1, p2, angle, length))
        return sorted(detected, key=lambda line: line.length, reverse=True)[
            : self.config.maximum_hough_lines
        ]

    def _line_candidates(
        self,
        lines: list[_DetectedLine],
        edge_map: np.ndarray,
        image_shape: tuple[int, int],
        scale_x: float,
        scale_y: float,
        scale_used: int,
        include_partial: bool,
    ) -> list[_Candidate]:
        if len(lines) < 2:
            return []
        target_ratio = self._target_ratio
        candidates: list[_Candidate] = []
        for theta in self._orientation_hypotheses(lines):
            u = np.asarray((cos(theta), sin(theta)), dtype=np.float32)
            v = np.asarray((-sin(theta), cos(theta)), dtype=np.float32)
            a_clusters = self._cluster_lines(lines, theta, u, v, image_shape)
            b_clusters = self._cluster_lines(lines, theta + pi / 2, v, u, image_shape)
            a_pairs = self._cluster_pairs(a_clusters, image_shape)
            b_pairs = self._cluster_pairs(b_clusters, image_shape)

            for a1, a2 in a_pairs[:14]:
                height = abs(a2.offset - a1.offset)
                for b1, b2 in b_pairs[:14]:
                    width = abs(b2.offset - b1.offset)
                    ratio = max(width, height) / max(1.0, min(width, height))
                    if ratio > target_ratio * 2.8:
                        continue
                    polygon = self._basis_polygon(u, v, a1.offset, a2.offset, b1.offset, b2.offset)
                    support = self._four_edge_line_support(a1, a2, b1, b2, width, height)
                    candidate = self._make_candidate(
                        polygon,
                        source="line4:hough",
                        scale_used=scale_used,
                        edge_map=edge_map,
                        image_shape=image_shape,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        geometry_score=0.96,
                        rectangularity_score=0.92,
                        line_support_score=support,
                        inferred_edges=0,
                    )
                    if candidate is not None:
                        candidates.append(candidate)

            if include_partial:
                candidates.extend(
                    self._partial_line_candidates(
                        u,
                        v,
                        a_clusters,
                        b_clusters,
                        a_pairs,
                        b_pairs,
                        edge_map,
                        image_shape,
                        scale_x,
                        scale_y,
                        scale_used,
                    )
                )
        return candidates

    def _orientation_hypotheses(self, lines: list[_DetectedLine]) -> list[float]:
        bin_count = 18
        histogram = np.zeros(bin_count, dtype=np.float64)
        for line in lines:
            folded = line.angle % (pi / 2)
            index = min(bin_count - 1, int(folded / (pi / 2) * bin_count))
            histogram[index] += line.length
        selected: list[float] = []
        for index in np.argsort(histogram)[::-1]:
            if histogram[index] <= 0:
                continue
            theta = (index + 0.5) * (pi / 2) / bin_count
            if all(self._angle_distance(theta, prior) > np.deg2rad(8) for prior in selected):
                selected.append(theta)
            if len(selected) == 4:
                break
        return selected

    def _cluster_lines(
        self,
        lines: list[_DetectedLine],
        expected_angle: float,
        direction: np.ndarray,
        normal: np.ndarray,
        image_shape: tuple[int, int],
    ) -> list[_LineCluster]:
        tolerance = np.deg2rad(self.config.line_angle_tolerance_degrees)
        entries: list[tuple[float, float, float, float]] = []
        for line in lines:
            if self._angle_distance(line.angle, expected_angle % pi) > tolerance:
                continue
            midpoint = (line.p1 + line.p2) / 2
            offset = float(np.dot(midpoint, normal))
            first = float(np.dot(line.p1, direction))
            second = float(np.dot(line.p2, direction))
            entries.append((offset, min(first, second), max(first, second), line.length))
        if not entries:
            return []
        entries.sort(key=lambda entry: entry[0])
        distance = max(
            4.0,
            min(image_shape) * self.config.line_cluster_distance_fraction,
        )
        grouped: list[list[tuple[float, float, float, float]]] = []
        for entry in entries:
            if not grouped or abs(entry[0] - np.mean([item[0] for item in grouped[-1]])) > distance:
                grouped.append([entry])
            else:
                grouped[-1].append(entry)
        clusters: list[_LineCluster] = []
        for group in grouped:
            weights = np.asarray([item[3] for item in group])
            offset = float(np.average([item[0] for item in group], weights=weights))
            intervals = [(item[1], item[2]) for item in group]
            merged_intervals = self._merge_intervals(intervals)
            coverage = sum(end - start for start, end in merged_intervals)
            clusters.append(
                _LineCluster(
                    offset=offset,
                    coverage=coverage,
                    projection_min=min(item[1] for item in group),
                    projection_max=max(item[2] for item in group),
                    segment_count=len(group),
                    intervals=merged_intervals,
                )
            )
        return sorted(clusters, key=lambda cluster: cluster.coverage, reverse=True)[
            : self.config.maximum_line_clusters
        ]

    def _cluster_pairs(
        self, clusters: list[_LineCluster], image_shape: tuple[int, int]
    ) -> list[tuple[_LineCluster, _LineCluster]]:
        minimum = min(image_shape) * self.config.line_min_separation_fraction
        pairs = [
            pair
            for pair in combinations(clusters, 2)
            if abs(pair[1].offset - pair[0].offset) >= minimum
        ]
        return sorted(
            pairs,
            key=lambda pair: pair[0].coverage + pair[1].coverage,
            reverse=True,
        )

    def _partial_line_candidates(
        self,
        u: np.ndarray,
        v: np.ndarray,
        a_clusters: list[_LineCluster],
        b_clusters: list[_LineCluster],
        a_pairs: list[tuple[_LineCluster, _LineCluster]],
        b_pairs: list[tuple[_LineCluster, _LineCluster]],
        edge_map: np.ndarray,
        image_shape: tuple[int, int],
        scale_x: float,
        scale_y: float,
        scale_used: int,
    ) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        target = self._target_ratio
        # Two parallel long edges plus one short edge: infer the opposite short edge.
        for a1, a2 in a_pairs[:10]:
            height = abs(a2.offset - a1.offset)
            expected_width = height * target
            for b1 in b_clusters[:6]:
                for sign in (-1.0, 1.0):
                    b2_offset = b1.offset + sign * expected_width
                    polygon = self._basis_polygon(
                        u, v, a1.offset, a2.offset, b1.offset, b2_offset
                    )
                    support = np.mean(
                        (
                            min(1.0, self._coverage_within(a1, b1.offset, b2_offset) / expected_width),
                            min(1.0, self._coverage_within(a2, b1.offset, b2_offset) / expected_width),
                            min(1.0, self._coverage_within(b1, a1.offset, a2.offset) / max(1.0, height)),
                            0.0,
                        )
                    )
                    candidate = self._make_candidate(
                        polygon,
                        source="partial3:hough",
                        scale_used=scale_used,
                        edge_map=edge_map,
                        image_shape=image_shape,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        geometry_score=0.78,
                        rectangularity_score=0.74,
                        line_support_score=float(support),
                        inferred_edges=1,
                    )
                    if candidate is not None:
                        candidates.append(candidate)

        # Two parallel short edges plus one long edge: infer the opposite long edge.
        for b1, b2 in b_pairs[:10]:
            width = abs(b2.offset - b1.offset)
            expected_height = width / target
            for a1 in a_clusters[:6]:
                for sign in (-1.0, 1.0):
                    a2_offset = a1.offset + sign * expected_height
                    polygon = self._basis_polygon(
                        u, v, a1.offset, a2_offset, b1.offset, b2.offset
                    )
                    support = np.mean(
                        (
                            min(1.0, self._coverage_within(a1, b1.offset, b2.offset) / max(1.0, width)),
                            0.0,
                            min(1.0, self._coverage_within(b1, a1.offset, a2_offset) / expected_height),
                            min(1.0, self._coverage_within(b2, a1.offset, a2_offset) / expected_height),
                        )
                    )
                    candidate = self._make_candidate(
                        polygon,
                        source="partial3:hough",
                        scale_used=scale_used,
                        edge_map=edge_map,
                        image_shape=image_shape,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        geometry_score=0.78,
                        rectangularity_score=0.74,
                        line_support_score=float(support),
                        inferred_edges=1,
                    )
                    if candidate is not None:
                        candidates.append(candidate)

        # Only opposite edges are reliable: infer extent from their joined segments.
        for a1, a2 in a_pairs[:10]:
            height = abs(a2.offset - a1.offset)
            expected_width = height * target
            projection_min = min(a1.projection_min, a2.projection_min)
            projection_max = max(a1.projection_max, a2.projection_max)
            center = (projection_min + projection_max) / 2
            b1_offset = center - expected_width / 2
            b2_offset = center + expected_width / 2
            polygon = self._basis_polygon(
                u, v, a1.offset, a2.offset, b1_offset, b2_offset
            )
            support = np.mean(
                (
                    min(1.0, self._coverage_within(a1, b1_offset, b2_offset) / expected_width),
                    min(1.0, self._coverage_within(a2, b1_offset, b2_offset) / expected_width),
                    0.0,
                    0.0,
                )
            )
            candidate = self._make_candidate(
                polygon,
                source="partial2:hough",
                scale_used=scale_used,
                edge_map=edge_map,
                image_shape=image_shape,
                scale_x=scale_x,
                scale_y=scale_y,
                geometry_score=0.62,
                rectangularity_score=0.62,
                line_support_score=float(support),
                inferred_edges=2,
            )
            if candidate is not None:
                candidates.append(candidate)
        for b1, b2 in b_pairs[:10]:
            width = abs(b2.offset - b1.offset)
            expected_height = width / target
            projection_min = min(b1.projection_min, b2.projection_min)
            projection_max = max(b1.projection_max, b2.projection_max)
            center = (projection_min + projection_max) / 2
            a1_offset = center - expected_height / 2
            a2_offset = center + expected_height / 2
            polygon = self._basis_polygon(
                u, v, a1_offset, a2_offset, b1.offset, b2.offset
            )
            support = np.mean(
                (
                    0.0,
                    0.0,
                    min(1.0, self._coverage_within(b1, a1_offset, a2_offset) / expected_height),
                    min(1.0, self._coverage_within(b2, a1_offset, a2_offset) / expected_height),
                )
            )
            candidate = self._make_candidate(
                polygon,
                source="partial2:hough",
                scale_used=scale_used,
                edge_map=edge_map,
                image_shape=image_shape,
                scale_x=scale_x,
                scale_y=scale_y,
                geometry_score=0.62,
                rectangularity_score=0.62,
                line_support_score=float(support),
                inferred_edges=2,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _basis_polygon(
        u: np.ndarray,
        v: np.ndarray,
        a1: float,
        a2: float,
        b1: float,
        b2: float,
    ) -> np.ndarray:
        return np.asarray(
            (
                b1 * u + a1 * v,
                b2 * u + a1 * v,
                b2 * u + a2 * v,
                b1 * u + a2 * v,
            ),
            dtype=np.float32,
        )

    @classmethod
    def _four_edge_line_support(
        cls,
        a1: _LineCluster,
        a2: _LineCluster,
        b1: _LineCluster,
        b2: _LineCluster,
        width: float,
        height: float,
    ) -> float:
        return float(
            np.mean(
                (
                    min(1.0, cls._coverage_within(a1, b1.offset, b2.offset) / max(1.0, width)),
                    min(1.0, cls._coverage_within(a2, b1.offset, b2.offset) / max(1.0, width)),
                    min(1.0, cls._coverage_within(b1, a1.offset, a2.offset) / max(1.0, height)),
                    min(1.0, cls._coverage_within(b2, a1.offset, a2.offset) / max(1.0, height)),
                )
            )
        )

    @staticmethod
    def _merge_intervals(
        intervals: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        if not intervals:
            return []
        merged: list[list[float]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1] + 3:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return [(start, end) for start, end in merged]

    @staticmethod
    def _coverage_within(
        cluster: _LineCluster, first: float, second: float
    ) -> float:
        start, end = sorted((first, second))
        return float(
            sum(
                max(0.0, min(interval_end, end) - max(interval_start, start))
                for interval_start, interval_end in cluster.intervals
            )
        )

    def _make_candidate(
        self,
        polygon: np.ndarray,
        source: str,
        scale_used: int,
        edge_map: np.ndarray,
        image_shape: tuple[int, int],
        scale_x: float,
        scale_y: float,
        geometry_score: float,
        rectangularity_score: float,
        line_support_score: float,
        inferred_edges: int,
    ) -> _Candidate | None:
        polygon = np.asarray(polygon, dtype=np.float32).reshape(4, 2)
        if not np.isfinite(polygon).all():
            return None
        height, width = image_shape
        raw_area = abs(float(cv2.contourArea(polygon)))
        if raw_area <= 0:
            return None
        visible = polygon.copy()
        visible[:, 0] = np.clip(visible[:, 0], 0, width - 1)
        visible[:, 1] = np.clip(visible[:, 1], 0, height - 1)
        visible_area = abs(float(cv2.contourArea(visible)))
        if visible_area / raw_area < 0.72:
            return None
        polygon = visible
        area = visible_area
        area_ratio = area / float(width * height)
        if not (
            self.config.minimum_area_ratio
            <= area_ratio
            <= self.config.maximum_area_ratio
        ):
            return None
        rotated = cv2.minAreaRect(polygon)
        rect_width, rect_height = rotated[1]
        if min(rect_width, rect_height) < self.config.contour_min_dimension_px:
            return None
        observed_ratio = self._perspective_tolerant_ratio(polygon)
        x, y, box_width, box_height = cv2.boundingRect(polygon.astype(np.int32))
        if (
            box_width >= width * self.config.whole_frame_width_ratio
            and box_height >= height * self.config.whole_frame_height_ratio
        ):
            return None
        border_touches = sum(
            (
                x <= 2,
                y <= 2,
                x + box_width >= width - 2,
                y + box_height >= height - 2,
            )
        )
        area_score = self._peaked_area_score(area_ratio)
        ratio_score = exp(
            -abs(log(observed_ratio / self._target_ratio))
            / self.config.ratio_log_tolerance
        )
        edge_score = self._edge_support(edge_map, polygon)
        interior_detail_score = self._interior_detail(edge_map, polygon)
        background_penalty = (
            self.config.plain_rectangle_penalty
            if interior_detail_score < 0.05
            and area_ratio > self.config.oversize_start_ratio
            and rectangularity_score > 0.88
            else 0.0
        )
        occlusion_penalty = (
            self.config.one_inferred_edge_penalty
            if inferred_edges == 1
            else self.config.two_inferred_edges_penalty
            if inferred_edges >= 2
            else 0.0
        )
        original_polygon = polygon.copy()
        original_polygon[:, 0] *= scale_x
        original_polygon[:, 1] *= scale_y
        candidate = _Candidate(
            polygon=original_polygon,
            source_methods={source},
            scales_used={scale_used},
            total_score=0.0,
            geometry_score=float(np.clip(geometry_score, 0.0, 1.0)),
            ratio_score=float(np.clip(ratio_score, 0.0, 1.0)),
            edge_score=edge_score,
            area_score=float(np.clip(area_score, 0.0, 1.0)),
            rectangularity_score=float(
                np.clip(rectangularity_score, 0.0, 1.0)
            ),
            line_support_score=float(np.clip(line_support_score, 0.0, 1.0)),
            agreement_score=0.0,
            interior_detail_score=interior_detail_score,
            interior_complexity_score=0.0,
            border_contrast_score=0.0,
            foreground_score=0.0,
            nested_candidate_score=0.0,
            background_penalty=background_penalty,
            oversize_penalty=0.0,
            uniform_surface_penalty=0.0,
            small_area_penalty=0.0,
            ratio_penalty=0.0,
            partial_evidence_bonus=0.0,
            occlusion_penalty=occlusion_penalty,
            inferred_edges=inferred_edges,
            area_ratio=area_ratio,
            observed_ratio=observed_ratio,
            border_touches=border_touches,
        )
        self._score_candidate(candidate)
        return candidate

    def _score_candidate(self, candidate: _Candidate) -> None:
        weights = self.config.weights
        # Full-card evidence is intentionally geometric.  Appearance remains a
        # secondary tie-breaker so a plain card back is not discarded in favor
        # of a detailed chip, portrait frame, logo, or printed rectangle.
        candidate.full_card_boundary_score = float(
            np.clip(
                0.29 * candidate.edge_score
                + 0.24 * candidate.line_support_score
                + 0.21 * candidate.geometry_score
                + 0.18 * candidate.ratio_score
                + 0.08 * candidate.agreement_score,
                0.0,
                1.0,
            )
        )
        if not candidate.contextual_scored:
            candidate.candidate_coverage_score = float(
                np.clip(
                    0.42 * candidate.area_score
                    + 0.32 * candidate.ratio_score
                    + 0.26 * candidate.geometry_score,
                    0.0,
                    1.0,
                )
            )
        candidate.total_score = float(
            np.clip(
                candidate.area_score * weights.area
                + candidate.geometry_score * weights.geometry
                + candidate.ratio_score * weights.aspect_ratio
                + candidate.edge_score * weights.edge_support
                + candidate.rectangularity_score * weights.rectangularity
                + candidate.line_support_score * weights.line_support
                + candidate.agreement_score * weights.method_agreement
                + candidate.interior_detail_score * weights.interior_detail
                + candidate.interior_complexity_score * weights.interior_complexity
                + candidate.border_contrast_score * weights.border_contrast
                + candidate.foreground_score * weights.foreground
                + candidate.nested_candidate_score * weights.nested_candidate
                - candidate.occlusion_penalty
                - candidate.background_penalty
                - candidate.oversize_penalty
                - candidate.uniform_surface_penalty
                - candidate.small_area_penalty
                - candidate.ratio_penalty
                + 0.06 * (candidate.full_card_boundary_score - 0.5)
                + 0.04 * (candidate.candidate_coverage_score - 0.5)
                - candidate.internal_subregion_penalty
                + candidate.partial_evidence_bonus
                - candidate.border_touches * self.config.border_touch_penalty,
                0.0,
                1.0,
            )
        )

    def _fuse_candidates(self, candidates: list[_Candidate]) -> list[_Candidate]:
        fused: list[_Candidate] = []
        for incoming in sorted(
            candidates, key=lambda candidate: candidate.total_score, reverse=True
        ):
            match = next(
                (
                    existing
                    for existing in fused
                    if self._same_geometry(existing, incoming)
                ),
                None,
            )
            if match is None:
                fused.append(self._copy_candidate(incoming))
                continue
            combined_methods = match.source_methods | incoming.source_methods
            combined_scales = match.scales_used | incoming.scales_used
            combined_line_support = max(
                match.line_support_score, incoming.line_support_score
            )
            # Scoring decides which fused candidate wins; it should not also
            # decide which polygon supplies the geometry.  A line-only quad
            # can score slightly higher while quantising an otherwise exact
            # contour angle, which made rotation jump in 2.5-degree steps.
            if self._candidate_anchor_quality(incoming) > self._candidate_anchor_quality(match):
                match.polygon = incoming.polygon.copy()
                match.geometry_score = incoming.geometry_score
                match.ratio_score = incoming.ratio_score
                match.edge_score = incoming.edge_score
                match.area_score = incoming.area_score
                match.rectangularity_score = incoming.rectangularity_score
                match.line_support_score = incoming.line_support_score
                match.interior_detail_score = incoming.interior_detail_score
                match.interior_complexity_score = incoming.interior_complexity_score
                match.border_contrast_score = incoming.border_contrast_score
                match.foreground_score = incoming.foreground_score
                match.nested_candidate_score = incoming.nested_candidate_score
                match.background_penalty = incoming.background_penalty
                match.oversize_penalty = incoming.oversize_penalty
                match.uniform_surface_penalty = incoming.uniform_surface_penalty
                match.small_area_penalty = incoming.small_area_penalty
                match.ratio_penalty = incoming.ratio_penalty
                match.partial_evidence_bonus = incoming.partial_evidence_bonus
                match.inferred_edges = incoming.inferred_edges
                match.occlusion_penalty = incoming.occlusion_penalty
                match.area_ratio = incoming.area_ratio
                match.observed_ratio = incoming.observed_ratio
                match.border_touches = incoming.border_touches
            match.source_methods = combined_methods
            match.scales_used = combined_scales
            combined_strategies = {
                method.split(":", 1)[0] for method in combined_methods
            }
            if (
                match.area_ratio <= self.config.oversize_start_ratio
                and {"contour", "rotated_rect"} & combined_strategies
                and {"line4", "partial3", "partial2"} & combined_strategies
            ):
                match.line_support_score = combined_line_support
            match.contextual_scored = False

        for candidate in fused:
            strategies = candidate.strategy_groups
            agreement = 0.0
            if "contour" in strategies and "rotated_rect" in strategies:
                agreement += 0.30
            line_groups = {"line4", "partial3", "partial2"} & strategies
            if line_groups and ({"contour", "rotated_rect"} & strategies):
                agreement += 0.55
            if len(candidate.scales_used) >= 2:
                agreement += 0.20
            if len(candidate.scales_used) >= 3:
                agreement += 0.10
            # Repeated extraction from independent threshold/edge views is
            # useful evidence even when every observation has the same broad
            # strategy label (for example several rotated-rect masks).
            agreement += min(0.20, max(0, len(candidate.source_methods) - 1) * 0.04)
            candidate.agreement_score = min(1.0, agreement)
            self._score_candidate(candidate)
        return sorted(fused, key=lambda candidate: candidate.total_score, reverse=True)

    @staticmethod
    def _candidate_anchor_quality(candidate: _Candidate) -> float:
        """Prefer directly observed contours when choosing fused geometry."""
        strategies = candidate.strategy_groups
        source_bonus = 0.0
        if "contour" in strategies:
            source_bonus = 0.10
        elif "rotated_rect" in strategies:
            source_bonus = 0.06
        elif "line4" in strategies:
            source_bonus = 0.02
        elif "partial3" in strategies:
            source_bonus = -0.04
        elif "partial2" in strategies:
            source_bonus = -0.08
        return float(
            0.30 * candidate.geometry_score
            + 0.28 * candidate.ratio_score
            + 0.24 * candidate.edge_score
            + 0.18 * candidate.rectangularity_score
            + source_bonus
        )

    def _contextual_rank(
        self, candidates: list[_Candidate], original_rgb: np.ndarray
    ) -> list[_Candidate]:
        """Run appearance and hierarchy scoring on retained Stage-A candidates."""
        retained = [
            self._copy_candidate(candidate)
            for candidate in candidates[: self.config.contextual_candidate_limit]
        ]
        if not retained:
            return []
        analysis, scale_x, scale_y = self._contextual_analysis_copy(original_rgb)
        gray = cv2.cvtColor(analysis, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(analysis, cv2.COLOR_RGB2LAB)
        edges = self._auto_canny(cv2.GaussianBlur(gray, (3, 3), 0.7))
        analysis_polygons: list[np.ndarray] = []
        rejection_sets: list[set[str]] = []
        for candidate in retained:
            polygon = candidate.polygon.copy().astype(np.float32)
            polygon[:, 0] *= scale_x
            polygon[:, 1] *= scale_y
            analysis_polygons.append(cv2.convexHull(polygon).reshape(-1, 2))
            complexity, border = self._appearance_metrics(
                gray, lab, edges, analysis_polygons[-1]
            )
            candidate.interior_complexity_score = complexity
            candidate.border_contrast_score = border
            candidate.foreground_score = float(
                np.clip(
                    0.52 * border
                    + 0.36 * complexity
                    + 0.12 * candidate.edge_score,
                    0.0,
                    1.0,
                )
            )
            candidate.nested_candidate_score = 0.0
            candidate.internal_subregion_penalty = 0.0
            candidate.oversize_penalty = self._oversize_penalty(candidate.area_ratio)
            candidate.uniform_surface_penalty = self._uniform_surface_penalty(
                candidate.area_ratio, complexity
            )
            small_progress = np.clip(
                (self.config.small_area_penalty_end_ratio - candidate.area_ratio)
                / (
                    self.config.small_area_penalty_end_ratio
                    - self.config.minimum_area_ratio
                ),
                0.0,
                1.0,
            )
            candidate.small_area_penalty = float(
                self.config.maximum_small_area_penalty
                * float(small_progress) ** 2
                * (3.0 - 2.0 * float(small_progress))
            )
            candidate.ratio_penalty = max(
                0.0, (0.38 - candidate.ratio_score) / 0.38
            ) * 0.09
            recovery_limit = 0.0
            if candidate.inferred_edges == 1:
                recovery_limit = self.config.one_edge_evidence_recovery
            elif candidate.inferred_edges >= 2:
                recovery_limit = self.config.two_edge_evidence_recovery
            candidate.partial_evidence_bonus = float(
                recovery_limit
                * min(1.0, candidate.line_support_score / 0.50)
                * candidate.ratio_score
                * candidate.edge_score
            )
            candidate.contextual_scored = True
            reasons: set[str] = set()
            if candidate.inferred_edges == 1 and border < 0.32:
                candidate.background_penalty += min(0.08, (0.32 - border) * 0.55)
                reasons.add("weak_local_boundary")
            if candidate.oversize_penalty > 0.035:
                reasons.add("oversized_background")
            if candidate.uniform_surface_penalty > 0.035:
                reasons.add("uniform_interior")
            if candidate.ratio_score < 0.35:
                reasons.add("poor_card_ratio")
            rejection_sets.append(reasons)

        contains: list[set[int]] = [set() for _ in retained]
        contained_by: list[set[int]] = [set() for _ in retained]
        overlaps: list[set[int]] = [set() for _ in retained]
        polygon_areas = [
            max(1.0, abs(float(cv2.contourArea(polygon))))
            for polygon in analysis_polygons
        ]
        for first_index, second_index in combinations(range(len(retained)), 2):
            intersection = self._convex_intersection_area(
                analysis_polygons[first_index], analysis_polygons[second_index]
            )
            if intersection <= 0:
                continue
            first_area = polygon_areas[first_index]
            second_area = polygon_areas[second_index]
            if first_area >= second_area:
                parent_index, child_index = first_index, second_index
                parent_area, child_area = first_area, second_area
            else:
                parent_index, child_index = second_index, first_index
                parent_area, child_area = second_area, first_area
            containment = intersection / child_area
            size_multiple = parent_area / child_area
            if (
                containment >= self.config.nested_containment_ratio
                and size_multiple >= self.config.nested_minimum_size_multiple
            ):
                contains[parent_index].add(child_index)
                contained_by[child_index].add(parent_index)
            elif intersection / min(first_area, second_area) >= 0.20:
                overlaps[first_index].add(second_index)
                overlaps[second_index].add(first_index)

        for parent_index, child_indices in enumerate(contains):
            parent = retained[parent_index]
            for child_index in child_indices:
                child = retained[child_index]
                size_multiple = polygon_areas[parent_index] / polygon_areas[child_index]
                size_score = float(np.clip(log(size_multiple) / log(10.0), 0.0, 1.0))
                ratio_advantage = max(0.0, child.ratio_score - parent.ratio_score)
                complexity_advantage = max(
                    0.0,
                    child.interior_complexity_score
                    - parent.interior_complexity_score,
                )
                child_strength = float(
                    np.clip(
                        0.52 * child.ratio_score
                        + 0.30 * child.interior_complexity_score
                        + 0.18 * child.border_contrast_score,
                        0.0,
                        1.0,
                    )
                )
                child.nested_candidate_score = max(
                    child.nested_candidate_score,
                    float(
                        np.clip(
                            0.32
                            + 0.24 * size_score
                            + 0.24 * ratio_advantage
                            + 0.20 * complexity_advantage,
                            0.0,
                            1.0,
                        )
                    ),
                )
                child.foreground_score = max(
                    child.foreground_score,
                    float(
                        np.clip(
                            0.42 * child.border_contrast_score
                            + 0.28 * child.interior_complexity_score
                            + 0.30
                            * max(ratio_advantage, complexity_advantage),
                            0.0,
                            1.0,
                        )
                    ),
                )
                parent_is_plausible_perimeter = (
                    parent.inferred_edges == 0
                    and bool(
                        {"contour", "rotated_rect"} & parent.strategy_groups
                    )
                    and parent.area_ratio <= self.config.oversize_start_ratio
                    and parent.geometry_score >= 0.72
                    and parent.ratio_score >= max(0.68, child.ratio_score - 0.16)
                    and parent.edge_score >= 0.62
                )
                if parent_is_plausible_perimeter:
                    parent.full_card_boundary_score = max(
                        parent.full_card_boundary_score,
                        float(
                            np.clip(
                                0.42 * parent.edge_score
                                + 0.26 * parent.ratio_score
                                + 0.20 * parent.line_support_score
                                + 0.12 * parent.geometry_score,
                                0.0,
                                1.0,
                            )
                        ),
                    )
                    parent.candidate_coverage_score = max(
                        parent.candidate_coverage_score,
                        float(np.clip(0.68 + 0.20 * size_score + 0.12 * parent.area_score, 0.0, 1.0)),
                    )
                    parent.nested_candidate_score = max(
                        parent.nested_candidate_score,
                        float(
                            np.clip(
                                0.24
                                + 0.24 * parent.ratio_score
                                + 0.32 * parent.edge_score
                                + 0.20 * size_score,
                                0.0,
                                1.0,
                            )
                        ),
                    )
                    child_fraction = polygon_areas[child_index] / max(
                        polygon_areas[parent_index], 1.0
                    )
                    if child_fraction <= 0.68:
                        inwardness = float(
                            np.clip((0.68 - child_fraction) / 0.50, 0.0, 1.0)
                        )
                        perimeter_advantage = max(
                            0.0,
                            parent.full_card_boundary_score
                            - child.full_card_boundary_score
                            + 0.12,
                        )
                        child.internal_subregion_penalty = max(
                            child.internal_subregion_penalty,
                            float(
                                np.clip(
                                    0.07
                                    + 0.10 * inwardness
                                    + 0.06 * perimeter_advantage,
                                    0.0,
                                    0.22,
                                )
                            ),
                        )
                        child.candidate_coverage_score = min(
                            child.candidate_coverage_score,
                            float(np.clip(child_fraction / 0.68, 0.0, 1.0)),
                        )
                        rejection_sets[child_index].add("internal_subregion")
                if (
                    child.inferred_edges == 0
                    and child.area_ratio >= self.config.area_plateau_min_ratio
                    and child_strength >= 0.46
                    and not parent_is_plausible_perimeter
                    and (
                        parent.area_ratio >= self.config.oversize_start_ratio
                        or parent.interior_complexity_score <= 0.55
                    )
                    and (
                        parent.area_ratio >= self.config.oversize_start_ratio
                        or ratio_advantage >= 0.08
                        or complexity_advantage >= 0.08
                    )
                ):
                    parent.oversize_penalty = min(
                        0.36,
                        parent.oversize_penalty
                        + self.config.nested_parent_penalty
                        * child_strength
                        * (0.62 + 0.38 * size_score),
                    )
                    rejection_sets[parent_index].add("nested_parent_penalty")

        # A reconstructed quad can accidentally combine a card edge with a
        # nearby mat/page edge.  If a complete overlapping candidate has a
        # comparable footprint and equally rich local appearance, ownership
        # belongs to that directly observed boundary.
        for inferred_index, candidate in enumerate(retained):
            if candidate.inferred_edges == 0:
                continue
            for direct_index in overlaps[inferred_index]:
                direct = retained[direct_index]
                size_ratio = direct.area_ratio / max(candidate.area_ratio, 1e-6)
                if (
                    direct.inferred_edges == 0
                    and 0.55 <= size_ratio <= 1.80
                    and direct.ratio_score >= 0.60
                    and direct.edge_score >= candidate.edge_score - 0.12
                    and (
                        direct.agreement_score >= 0.20
                        or direct.line_support_score >= 0.45
                    )
                ):
                    candidate.background_penalty += 0.15
                    rejection_sets[inferred_index].add("weaker_inferred_geometry")
                    direct.foreground_score = max(direct.foreground_score, 0.72)
                    direct.nested_candidate_score = max(
                        direct.nested_candidate_score, 0.70
                    )
                    break

        complete_candidates = [
            (index, candidate)
            for index, candidate in enumerate(retained)
            if candidate.inferred_edges == 0
            and candidate.area_ratio >= self.config.area_plateau_min_ratio
            and candidate.ratio_score >= 0.60
            and candidate.edge_score >= 0.65
        ]
        if complete_candidates:
            complete_index, complete = max(
                complete_candidates,
                key=lambda item: (
                    0.28 * item[1].ratio_score
                    + 0.22 * item[1].edge_score
                    + 0.18 * item[1].geometry_score
                    + 0.12
                    * min(1.0, max(0, len(item[1].source_methods) - 1) / 3.0)
                    + 0.08 * item[1].border_contrast_score
                    + 0.07 * item[1].area_score
                    + 0.05 * item[1].interior_complexity_score
                ),
            )
            for inferred_index, candidate in enumerate(retained):
                if candidate.inferred_edges == 0:
                    continue
                size_ratio = complete.area_ratio / max(candidate.area_ratio, 1e-6)
                if (
                    0.45 <= size_ratio <= 3.0
                    and complete.geometry_score >= candidate.geometry_score - 0.12
                    and complete.ratio_score >= candidate.ratio_score - 0.15
                ):
                    candidate.background_penalty += 0.09
                    rejection_sets[inferred_index].add("complete_candidate_preferred")
                    complete.foreground_score = max(complete.foreground_score, 0.85)
                    complete.nested_candidate_score = max(
                        complete.nested_candidate_score, 0.90
                    )

        for candidate in retained:
            self._score_candidate(candidate)
        ranked = sorted(retained, key=lambda candidate: candidate.total_score, reverse=True)
        final_ids = {id(candidate): f"C{index}" for index, candidate in enumerate(ranked, 1)}
        original_index = {id(candidate): index for index, candidate in enumerate(retained)}
        for candidate in ranked:
            index = original_index[id(candidate)]
            candidate.candidate_id = final_ids[id(candidate)]
            candidate.contains_candidates = tuple(
                final_ids[id(retained[other])] for other in sorted(contains[index])
            )
            candidate.contained_by_candidates = tuple(
                final_ids[id(retained[other])] for other in sorted(contained_by[index])
            )
            candidate.overlap_candidates = tuple(
                final_ids[id(retained[other])] for other in sorted(overlaps[index])
            )
            candidate.rejection_reasons = tuple(sorted(rejection_sets[index]))
        return ranked

    def _appearance_metrics(
        self,
        gray: np.ndarray,
        lab: np.ndarray,
        edges: np.ndarray,
        polygon: np.ndarray,
    ) -> tuple[float, float]:
        height, width = gray.shape
        center = np.mean(polygon, axis=0)
        band = self.config.appearance_border_band_fraction
        inner_polygon = center + (polygon - center) * (1.0 - band)
        detail_polygon = center + (polygon - center) * 0.78
        outer_polygon = center + (polygon - center) * (1.0 + band)
        minimum = np.floor(outer_polygon.min(axis=0) - 2).astype(int)
        maximum = np.ceil(outer_polygon.max(axis=0) + 2).astype(int)
        left = max(0, minimum[0])
        top = max(0, minimum[1])
        right = min(width, maximum[0] + 1)
        bottom = min(height, maximum[1] + 1)
        if right - left < 8 or bottom - top < 8:
            return 0.0, 0.0
        local_polygon = polygon - (left, top)
        local_inner = inner_polygon - (left, top)
        local_detail = detail_polygon - (left, top)
        local_outer = outer_polygon - (left, top)
        shape = (bottom - top, right - left)
        card_mask = np.zeros(shape, dtype=np.uint8)
        inner_mask = np.zeros(shape, dtype=np.uint8)
        detail_mask = np.zeros(shape, dtype=np.uint8)
        outer_mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillConvexPoly(card_mask, np.round(local_polygon).astype(np.int32), 255)
        cv2.fillConvexPoly(inner_mask, np.round(local_inner).astype(np.int32), 255)
        cv2.fillConvexPoly(detail_mask, np.round(local_detail).astype(np.int32), 255)
        cv2.fillConvexPoly(outer_mask, np.round(local_outer).astype(np.int32), 255)
        local_gray = gray[top:bottom, left:right]
        local_lab = lab[top:bottom, left:right]
        local_edges = edges[top:bottom, left:right]
        detail_pixels = detail_mask > 0
        if int(detail_pixels.sum()) < 30:
            return 0.0, 0.0
        edge_density = float(np.count_nonzero(local_edges[detail_pixels])) / float(
            detail_pixels.sum()
        )
        gray_values = local_gray[detail_pixels]
        lab_values = local_lab[detail_pixels]
        gray_variation = float(np.clip(np.std(gray_values) / 48.0, 0.0, 1.0))
        color_variation = float(
            np.clip(
                (np.std(lab_values[:, 1]) + np.std(lab_values[:, 2])) / 42.0,
                0.0,
                1.0,
            )
        )
        histogram = np.histogram(gray_values, bins=32, range=(0, 256))[0].astype(float)
        probabilities = histogram[histogram > 0] / max(1.0, histogram.sum())
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))
        entropy_score = float(np.clip(entropy / 4.5, 0.0, 1.0))
        edge_density_score = float(np.clip((edge_density - 0.0015) / 0.045, 0.0, 1.0))
        complexity = float(
            np.clip(
                0.34 * edge_density_score
                + 0.22 * gray_variation
                + 0.20 * color_variation
                + 0.24 * entropy_score,
                0.0,
                1.0,
            )
        )

        inside_band = (card_mask > 0) & (inner_mask == 0)
        outside_band = (outer_mask > 0) & (card_mask == 0)
        if int(inside_band.sum()) < 12 or int(outside_band.sum()) < 12:
            return complexity, 0.0
        inside_mean = np.mean(local_lab[inside_band].astype(np.float32), axis=0)
        outside_mean = np.mean(local_lab[outside_band].astype(np.float32), axis=0)
        color_transition = float(
            np.clip(np.linalg.norm(inside_mean - outside_mean) / 72.0, 0.0, 1.0)
        )
        boundary_mask = inside_band | outside_band
        boundary_density = float(np.count_nonzero(local_edges[boundary_mask])) / float(
            boundary_mask.sum()
        )
        border_contrast = float(
            np.clip(
                0.58 * color_transition
                + 0.42 * np.clip(boundary_density / 0.13, 0.0, 1.0),
                0.0,
                1.0,
            )
        )
        return complexity, border_contrast

    def _peaked_area_score(self, area_ratio: float) -> float:
        if area_ratio < self.config.area_plateau_min_ratio:
            return float(
                np.clip(
                    0.18
                    + 0.62
                    * np.sqrt(area_ratio / self.config.area_plateau_min_ratio),
                    0.0,
                    0.80,
                )
            )
        if area_ratio <= self.config.area_plateau_max_ratio:
            distance = abs(log(area_ratio / self.config.preferred_area_ratio))
            return float(np.clip(0.80 + 0.20 * exp(-(distance**2) / 1.8), 0.0, 1.0))
        span = max(
            0.01,
            self.config.maximum_area_ratio - self.config.area_plateau_max_ratio,
        )
        decline = (area_ratio - self.config.area_plateau_max_ratio) / span
        return float(np.clip(0.80 * exp(-2.3 * decline), 0.08, 0.80))

    def _oversize_penalty(self, area_ratio: float) -> float:
        if area_ratio <= self.config.oversize_start_ratio:
            return 0.0
        progress = float(
            np.clip(
                (area_ratio - self.config.oversize_start_ratio)
                / (self.config.oversize_full_ratio - self.config.oversize_start_ratio),
                0.0,
                1.0,
            )
        )
        smooth = progress * progress * (3.0 - 2.0 * progress)
        return self.config.maximum_oversize_penalty * smooth

    def _uniform_surface_penalty(
        self, area_ratio: float, complexity: float
    ) -> float:
        threshold = self.config.uniform_complexity_threshold
        if complexity >= threshold or area_ratio < self.config.oversize_start_ratio:
            return 0.0
        uniformity = (threshold - complexity) / threshold
        area_factor = float(
            np.clip(
                (area_ratio - self.config.oversize_start_ratio)
                / (
                    self.config.oversize_full_ratio
                    - self.config.oversize_start_ratio
                ),
                0.0,
                1.0,
            )
        )
        return self.config.maximum_uniform_penalty * uniformity * area_factor

    @staticmethod
    def _perspective_tolerant_ratio(polygon: np.ndarray) -> float:
        points = np.asarray(polygon, dtype=np.float32).reshape(4, 2)
        lengths = np.asarray(
            [
                np.linalg.norm(points[(index + 1) % 4] - points[index])
                for index in range(4)
            ],
            dtype=np.float64,
        )
        first_pair = (lengths[0] + lengths[2]) / 2.0
        second_pair = (lengths[1] + lengths[3]) / 2.0
        return float(max(first_pair, second_pair) / max(1.0, min(first_pair, second_pair)))

    def _contextual_analysis_copy(
        self, original: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        height, width = original.shape[:2]
        largest = max(width, height)
        if largest <= self.config.contextual_analysis_long_edge:
            return original, 1.0, 1.0
        factor = self.config.contextual_analysis_long_edge / largest
        resized = cv2.resize(
            original,
            (max(1, int(round(width * factor))), max(1, int(round(height * factor)))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, resized.shape[1] / width, resized.shape[0] / height

    @staticmethod
    def _convex_intersection_area(first: np.ndarray, second: np.ndarray) -> float:
        try:
            area, _ = cv2.intersectConvexConvex(
                cv2.convexHull(first.astype(np.float32)),
                cv2.convexHull(second.astype(np.float32)),
            )
            return float(max(0.0, area))
        except cv2.error:
            return 0.0

    @staticmethod
    def _copy_candidate(candidate: _Candidate) -> _Candidate:
        return _Candidate(
            polygon=candidate.polygon.copy(),
            source_methods=set(candidate.source_methods),
            scales_used=set(candidate.scales_used),
            total_score=candidate.total_score,
            geometry_score=candidate.geometry_score,
            ratio_score=candidate.ratio_score,
            edge_score=candidate.edge_score,
            area_score=candidate.area_score,
            rectangularity_score=candidate.rectangularity_score,
            line_support_score=candidate.line_support_score,
            agreement_score=candidate.agreement_score,
            interior_detail_score=candidate.interior_detail_score,
            interior_complexity_score=candidate.interior_complexity_score,
            border_contrast_score=candidate.border_contrast_score,
            foreground_score=candidate.foreground_score,
            nested_candidate_score=candidate.nested_candidate_score,
            background_penalty=candidate.background_penalty,
            oversize_penalty=candidate.oversize_penalty,
            uniform_surface_penalty=candidate.uniform_surface_penalty,
            small_area_penalty=candidate.small_area_penalty,
            ratio_penalty=candidate.ratio_penalty,
            partial_evidence_bonus=candidate.partial_evidence_bonus,
            occlusion_penalty=candidate.occlusion_penalty,
            inferred_edges=candidate.inferred_edges,
            area_ratio=candidate.area_ratio,
            observed_ratio=candidate.observed_ratio,
            border_touches=candidate.border_touches,
            full_card_boundary_score=candidate.full_card_boundary_score,
            candidate_coverage_score=candidate.candidate_coverage_score,
            internal_subregion_penalty=candidate.internal_subregion_penalty,
            candidate_id=candidate.candidate_id,
            contains_candidates=tuple(candidate.contains_candidates),
            contained_by_candidates=tuple(candidate.contained_by_candidates),
            overlap_candidates=tuple(candidate.overlap_candidates),
            rejection_reasons=tuple(candidate.rejection_reasons),
            contextual_scored=candidate.contextual_scored,
        )

    def _same_geometry(self, first: _Candidate, second: _Candidate) -> bool:
        if abs(log(first.observed_ratio / second.observed_ratio)) > 0.28:
            return False
        iou = self._intersection_over_union(first.bounding_box, second.bounding_box)
        if iou >= 0.64:
            return True
        first_center = np.mean(first.polygon, axis=0)
        second_center = np.mean(second.polygon, axis=0)
        first_box = first.bounding_box
        second_box = second.bounding_box
        first_diagonal = np.hypot(
            first_box[2] - first_box[0], first_box[3] - first_box[1]
        )
        second_diagonal = np.hypot(
            second_box[2] - second_box[0], second_box[3] - second_box[1]
        )
        center_distance = float(np.linalg.norm(first_center - second_center))
        size_ratio = max(first_diagonal, second_diagonal) / max(
            1.0, min(first_diagonal, second_diagonal)
        )
        return center_distance < min(first_diagonal, second_diagonal) * 0.10 and size_ratio < 1.28

    def _confidence(
        self, best: _Candidate, second: _Candidate | None
    ) -> tuple[float, dict[str, Any]]:
        if (
            second is not None
            and second.area_ratio < self.config.tiny_area_ratio
            and best.area_ratio >= self.config.tiny_area_ratio * 2
        ):
            second = None
        margin = best.total_score - second.total_score if second is not None else 1.0
        if second is None:
            ambiguity_penalty = 0.0
            margin_bonus = 0.05
        elif margin < self.config.ambiguity_close_gap:
            ambiguity_penalty = self.config.ambiguity_close_penalty
            margin_bonus = 0.0
        elif margin < self.config.ambiguity_medium_gap:
            ambiguity_penalty = self.config.ambiguity_medium_penalty
            margin_bonus = 0.0
        elif margin < self.config.ambiguity_wide_gap:
            ambiguity_penalty = self.config.ambiguity_wide_penalty
            margin_bonus = 0.02
        else:
            ambiguity_penalty = 0.0
            margin_bonus = min(0.07, margin * 0.22)
        confidence = best.total_score - ambiguity_penalty + margin_bonus
        ambiguity_review_floor = False
        if (
            ambiguity_penalty >= self.config.ambiguity_close_penalty
            and best.total_score >= 0.60
            and best.ratio_score >= 0.60
            and best.edge_score >= 0.75
        ):
            # Two strong, genuinely card-like regions are a review case, not
            # evidence that neither region exists.
            confidence = max(confidence, self.config.medium_confidence)
            # A near-tie between two distinct, fully card-like regions must
            # remain a review case even when both absolute scores are high.
            confidence = min(confidence, self.config.high_confidence - 0.01)
            ambiguity_review_floor = True
        if best.area_ratio < self.config.tiny_area_ratio:
            confidence = min(confidence - 0.08, 0.52)
        if best.inferred_edges == 1:
            confidence = min(confidence, 0.72)
        elif best.inferred_edges >= 2:
            confidence = min(confidence, 0.62)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        return confidence, {
            "absolute_quality": round(best.total_score, 4),
            "method_agreement": round(best.agreement_score, 4),
            "margin_over_second": round(margin, 4),
            "margin_bonus": round(margin_bonus, 4),
            "ambiguity_penalty": round(ambiguity_penalty, 4),
            "ambiguity_review_floor": ambiguity_review_floor,
            "inferred_edges": best.inferred_edges,
            "final_confidence": round(confidence, 4),
        }

    def _can_early_accept(self, candidates: list[_Candidate]) -> bool:
        if not candidates or candidates[0].total_score < self.config.early_accept_score:
            return False
        margin = (
            candidates[0].total_score - candidates[1].total_score
            if len(candidates) > 1
            else 1.0
        )
        return margin >= self.config.early_accept_margin

    def _needs_reconstruction(self, candidates: list[_Candidate]) -> bool:
        if not candidates:
            return True
        if candidates[0].total_score < self.config.high_confidence:
            return True
        if candidates[0].edge_score < 0.48:
            return True
        if len(candidates) > 1 and candidates[0].total_score - candidates[1].total_score < 0.07:
            return True
        return False

    def _looks_already_cropped(
        self,
        pass_result: _PassResult,
        candidates: list[_Candidate],
        original_width: int,
        original_height: int,
    ) -> bool:
        height, width = pass_result.working_rgb.shape[:2]
        canvas_ratio = max(width, height) / min(width, height)
        if abs(log(canvas_ratio / self._target_ratio)) > 0.065:
            return False
        # A PDF frame preprocessor (or a tightly captured image) can leave a
        # small safety band around an otherwise full-card raster.  A large,
        # ratio-correct contour inset on every side is direct full-card
        # evidence even when a low-texture back has almost no internal edges.
        for candidate in candidates[:5]:
            left, top, right, bottom = candidate.bounding_box
            near_perimeter = (
                left <= original_width * 0.07
                and top <= original_height * 0.07
                and right >= original_width * 0.93
                and bottom >= original_height * 0.93
            )
            if (
                candidate.area_ratio >= 0.78
                and candidate.geometry_score >= 0.85
                and candidate.ratio_score >= 0.85
                and candidate.rectangularity_score >= 0.85
                and near_perimeter
            ):
                return True
        corner_size = max(3, int(round(min(width, height) * 0.035)))
        corner_samples = np.asarray(
            [
                pass_result.working_rgb[:corner_size, :corner_size].mean(axis=(0, 1)),
                pass_result.working_rgb[:corner_size, -corner_size:].mean(axis=(0, 1)),
                pass_result.working_rgb[-corner_size:, -corner_size:].mean(axis=(0, 1)),
                pass_result.working_rgb[-corner_size:, :corner_size].mean(axis=(0, 1)),
            ],
            dtype=np.float32,
        )
        corner_color = corner_samples.mean(axis=0)
        corner_consistency = float(
            np.linalg.norm(corner_samples - corner_color, axis=1).max()
        )
        center = pass_result.working_rgb[
            height // 4 : height * 3 // 4,
            width // 4 : width * 3 // 4,
        ]
        center_color = np.median(center.reshape(-1, 3), axis=0)
        corner_hsv = cv2.cvtColor(
            np.uint8([[np.clip(corner_color, 0, 255)]]), cv2.COLOR_RGB2HSV
        )[0, 0]
        if (
            corner_consistency <= 12.0
            and int(corner_hsv[1]) <= 55
            and int(corner_hsv[2]) >= 180
            and float(np.linalg.norm(center_color - corner_color)) >= 28.0
        ):
            return True
        if candidates and candidates[0].area_ratio >= 0.075:
            return False
        if any(
            candidate.area_ratio >= 0.01
            and candidate.ratio_score >= 0.78
            and candidate.geometry_score >= 0.65
            for candidate in candidates[:10]
        ):
            return False
        inset_x = max(4, int(round(width * 0.06)))
        inset_y = max(4, int(round(height * 0.06)))
        interior = pass_result.working_rgb[
            inset_y : height - inset_y, inset_x : width - inset_x
        ]
        interior_edges = pass_result.edge_map[
            inset_y : height - inset_y, inset_x : width - inset_x
        ]
        if interior.size == 0:
            return False
        edge_density = cv2.countNonZero(interior_edges) / float(interior_edges.size)
        return edge_density >= 0.012 and float(np.std(interior)) >= 18.0

    def _attach_pass_debug(
        self,
        debug_info: dict[str, Any],
        passes: list[_PassResult],
        candidates: list[_Candidate],
        original: np.ndarray,
    ) -> None:
        debug_info["passes"] = [
            {
                "name": result.name,
                "scale_used": result.scale_used,
                "candidate_count": len(result.candidates),
                "line_segment_count": len(result.line_segments),
            }
            for result in passes
        ]
        debug_info["scale_used"] = [result.scale_used for result in passes]
        if passes:
            debug_info["working_size"] = (
                passes[0].working_rgb.shape[1],
                passes[0].working_rgb.shape[0],
            )
        if not self.debug:
            return
        stage_images: dict[str, Image.Image] = {
            "original": Image.fromarray(original).copy()
        }
        for result in passes:
            stage_images[f"{result.name}_working"] = Image.fromarray(
                result.working_rgb
            ).copy()
            stage_images[f"{result.name}_edges"] = Image.fromarray(
                result.edge_map
            ).copy()
            line_overlay = result.working_rgb.copy()
            for line in result.line_segments:
                cv2.line(
                    line_overlay,
                    tuple(line.p1.astype(int)),
                    tuple(line.p2.astype(int)),
                    (245, 180, 35),
                    2,
                )
            stage_images[f"{result.name}_lines"] = Image.fromarray(
                line_overlay
            ).copy()
            for stage_name, stage in result.stage_images.items():
                stage_images[f"{result.name}_{stage_name}"] = Image.fromarray(
                    stage
                ).copy()
        richest_pass = passes[-1]
        for alias in ("grayscale", "otsu", "adaptive"):
            if alias in richest_pass.stage_images:
                stage_images[alias] = Image.fromarray(
                    richest_pass.stage_images[alias]
                ).copy()
        stage_images["edges"] = Image.fromarray(richest_pass.edge_map).copy()
        overlay, overlay_scale = self._debug_overlay_canvas(original)
        for index, candidate in enumerate(candidates[:20], start=1):
            points = (candidate.polygon * overlay_scale).astype(np.int32)
            color = (45, 205, 95) if index == 1 else (230, 155, 35) if index == 2 else (70, 135, 230)
            cv2.polylines(overlay, [points], True, color, 2 if index > 1 else 4)
            anchor = tuple(points[0])
            cv2.putText(
                overlay,
                (
                    f"{candidate.candidate_id or f'C{index}'} "
                    f"T{candidate.total_score:.2f} R{candidate.ratio_score:.2f} "
                    f"A{candidate.area_score:.2f}"
                ),
                anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
            detail_anchor = (anchor[0], anchor[1] + 16)
            cv2.putText(
                overlay,
                (
                    f"O{candidate.oversize_penalty:.2f} "
                    f"I{candidate.interior_complexity_score:.2f} "
                    f"N{candidate.nested_candidate_score:.2f}"
                ),
                detail_anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                color,
                1,
                cv2.LINE_AA,
            )
        stage_images["candidate_scores"] = Image.fromarray(overlay).copy()
        stage_images["candidate_overlay"] = stage_images["candidate_scores"].copy()
        debug_info["stage_images"] = stage_images

    @staticmethod
    def _debug_overlay_canvas(original: np.ndarray) -> tuple[np.ndarray, float]:
        largest = max(original.shape[:2])
        if largest <= 1400:
            return original.copy(), 1.0
        scale = 1400 / largest
        resized = cv2.resize(
            original,
            (
                int(round(original.shape[1] * scale)),
                int(round(original.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    def _finalize(
        self,
        result: CardDetectionResult,
        started: float,
        debug_info: dict[str, Any],
    ) -> CardDetectionResult:
        processing_time_ms = (perf_counter() - started) * 1000
        debug_info["processing_time_ms"] = round(processing_time_ms, 2)
        debug_info["confidence"] = round(result.confidence, 4)
        debug_info["selected_method"] = result.method
        result.debug_info.update(debug_info)
        LOGGER.debug(
            "card_detection %s",
            {
                "candidate_count": debug_info.get("candidate_count", 0),
                "best_score": debug_info.get("best_score"),
                "second_best_score": debug_info.get("second_best_score"),
                "confidence": round(result.confidence, 4),
                "selected_method": result.method,
                "inferred_edge_count": debug_info.get("selected_candidate", {}).get(
                    "inferred_edges", 0
                ),
                "scale_used": debug_info.get("scale_used", []),
                "processing_time_ms": round(processing_time_ms, 2),
            },
        )
        return result

    @property
    def _target_ratio(self) -> float:
        return max(
            self.card_size.width_mm / self.card_size.height_mm,
            self.card_size.height_mm / self.card_size.width_mm,
        )

    def _edge_support(self, edges: np.ndarray, polygon: np.ndarray) -> float:
        mask = np.zeros_like(edges)
        thickness = max(
            2,
            int(round(min(edges.shape) * self.config.edge_band_fraction)),
        )
        cv2.polylines(mask, [polygon.astype(np.int32)], True, 255, thickness)
        boundary_pixels = cv2.countNonZero(mask)
        if boundary_pixels == 0:
            return 0.0
        supported = cv2.countNonZero(cv2.bitwise_and(edges, mask))
        return float(np.clip((supported / boundary_pixels) / 0.18, 0.0, 1.0))

    @staticmethod
    def _interior_detail(edges: np.ndarray, polygon: np.ndarray) -> float:
        center = np.mean(polygon, axis=0)
        inner = center + (polygon - center) * 0.76
        mask = np.zeros_like(edges)
        cv2.fillConvexPoly(mask, inner.astype(np.int32), 255)
        interior_pixels = cv2.countNonZero(mask)
        if interior_pixels == 0:
            return 0.0
        detail_pixels = cv2.countNonZero(cv2.bitwise_and(edges, mask))
        density = detail_pixels / float(interior_pixels)
        return float(np.clip(density / 0.006, 0.0, 1.0))

    @staticmethod
    def _angle_score(approximation: np.ndarray) -> float:
        points = approximation.reshape(-1, 2).astype(np.float32)
        if len(points) != 4:
            return 0.0
        scores: list[float] = []
        for index in range(4):
            previous = points[(index - 1) % 4] - points[index]
            following = points[(index + 1) % 4] - points[index]
            denominator = float(np.linalg.norm(previous) * np.linalg.norm(following))
            if denominator == 0:
                scores.append(0.0)
                continue
            cosine = abs(float(np.dot(previous, following) / denominator))
            scores.append(max(0.0, 1.0 - cosine / 0.58))
        return float(np.mean(scores))

    @staticmethod
    def _angle_distance(first: float, second: float) -> float:
        difference = abs((first - second) % pi)
        return min(difference, pi - difference)

    @staticmethod
    def _intersection_over_union(
        first: tuple[int, int, int, int], second: tuple[int, int, int, int]
    ) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        if intersection == 0:
            return 0.0
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        second_area = (second[2] - second[0]) * (second[3] - second[1])
        return intersection / float(first_area + second_area - intersection)

    @staticmethod
    def _long_edge_angle(points: np.ndarray) -> float:
        rotated = cv2.minAreaRect(points.astype(np.float32))
        box = cv2.boxPoints(rotated)
        edges = np.roll(box, -1, axis=0) - box
        longest = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
        angle = degrees(atan2(float(longest[1]), float(longest[0])))
        while angle <= -90:
            angle += 180
        while angle > 90:
            angle -= 180
        return angle

    def _rotated_crop(
        self, original_rgb: np.ndarray, polygon: np.ndarray
    ) -> tuple[Image.Image, float]:
        height, width = original_rgb.shape[:2]
        angle = self._long_edge_angle(polygon)
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cosine_value = abs(matrix[0, 0])
        sine_value = abs(matrix[0, 1])
        output_width = int(round(height * sine_value + width * cosine_value))
        output_height = int(round(height * cosine_value + width * sine_value))
        matrix[0, 2] += output_width / 2.0 - center[0]
        matrix[1, 2] += output_height / 2.0 - center[1]
        rotated = cv2.warpAffine(
            original_rgb,
            matrix,
            (output_width, output_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        transformed = cv2.transform(polygon[None, :, :], matrix)[0]
        minimum = transformed.min(axis=0)
        maximum = transformed.max(axis=0)
        margin_x = max(
            2.0,
            (maximum[0] - minimum[0])
            * self.config.crop_safety_margin_fraction,
        )
        margin_y = max(
            2.0,
            (maximum[1] - minimum[1])
            * self.config.crop_safety_margin_fraction,
        )
        left = max(0, int(np.floor(minimum[0] - margin_x)))
        top = max(0, int(np.floor(minimum[1] - margin_y)))
        right = min(output_width, int(np.ceil(maximum[0] + margin_x)))
        bottom = min(output_height, int(np.ceil(maximum[1] + margin_y)))
        crop = Image.fromarray(rotated[top:bottom, left:right]).copy()
        if crop.height > crop.width:
            crop = crop.transpose(Image.Transpose.ROTATE_90)
            angle = angle + 90 if angle <= 0 else angle - 90
        return crop, angle

    @staticmethod
    def _axis_aligned_bounds(
        polygon: np.ndarray, width: int, height: int
    ) -> tuple[int, int, int, int]:
        return (
            max(0, int(np.floor(polygon[:, 0].min()))),
            max(0, int(np.floor(polygon[:, 1].min()))),
            min(width, int(np.ceil(polygon[:, 0].max()))),
            min(height, int(np.ceil(polygon[:, 1].max()))),
        )

    @staticmethod
    def _polygon_tuple(polygon: np.ndarray) -> tuple[tuple[float, float], ...]:
        return tuple((float(point[0]), float(point[1])) for point in polygon)

    @staticmethod
    def _failure(reason: str) -> CardDetectionResult:
        return CardDetectionResult(
            success=False,
            confidence=0.0,
            confidence_level="none",
            method="hybrid",
            debug_info={"reason": reason},
        )
