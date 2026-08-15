from __future__ import annotations

import logging
from dataclasses import dataclass
from math import acos, degrees, log

import cv2
import numpy as np
from PIL import Image

from cardlayout.models.detection import Point
from cardlayout.models.perspective import (
    CornerRefinementResult,
    EdgeFitResult,
    Line,
    PerspectiveConfig,
)

LOGGER = logging.getLogger(__name__)
EDGE_NAMES = ("top", "right", "bottom", "left")


@dataclass(slots=True)
class _EdgeDebug:
    band_polygon: np.ndarray
    candidates: np.ndarray
    inliers: np.ndarray
    expected_start: np.ndarray
    expected_end: np.ndarray
    evidence: np.ndarray


class PreciseCornerRefiner:
    """Fit physical card boundaries inside a local high-resolution ROI."""

    def __init__(
        self,
        config: PerspectiveConfig | None = None,
        debug: bool = False,
        target_ratio: float | None = None,
    ) -> None:
        self.config = config or PerspectiveConfig()
        self.debug = debug
        self.target_ratio = target_ratio

    def refine(
        self,
        rgb: np.ndarray,
        rough_corners: np.ndarray,
        detector_inferred_edges: int = 0,
    ) -> CornerRefinementResult:
        rough = np.asarray(rough_corners, dtype=np.float32)
        rough_tuple = self._point_tuple(rough)
        if rgb.ndim != 3 or rough.shape != (4, 2):
            return self._fallback(rough_tuple, "Invalid refinement input.")

        roi_box = self._roi_box(rough, rgb.shape[1], rgb.shape[0])
        left, top, right, bottom = roi_box
        roi = rgb[top:bottom, left:right].copy()
        if roi.size == 0:
            return self._fallback(rough_tuple, "The local card region is empty.", roi_box)
        scale = min(
            1.0,
            self.config.refinement_max_roi_long_edge / max(roi.shape[:2]),
        )
        if scale < 1.0:
            roi = cv2.resize(
                roi,
                (
                    max(2, int(round(roi.shape[1] * scale))),
                    max(2, int(round(roi.shape[0] * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        local_rough = rough.copy()
        local_rough[:, 0] = (local_rough[:, 0] - left) * scale
        local_rough[:, 1] = (local_rough[:, 1] - top) * scale
        short_side = self._short_side(local_rough)
        coarse_band = float(np.clip(
            short_side * self.config.coarse_search_band_fraction, 4.0, 90.0
        ))
        fine_band = float(np.clip(
            short_side * self.config.fine_search_band_fraction, 2.5, 36.0
        ))
        representations = self._representations(roi)

        fitted: list[EdgeFitResult] = []
        edge_debug: dict[str, _EdgeDebug] = {}
        for index, name in enumerate(EDGE_NAMES):
            start = local_rough[index]
            end = local_rough[(index + 1) % 4]
            coarse, coarse_debug = self._fit_edge_from_image(
                name,
                representations,
                start,
                end,
                coarse_band,
                pass_number=1,
            )
            chosen = coarse
            chosen_debug = coarse_debug
            if coarse.success and coarse.line is not None:
                coarse_line = np.asarray(coarse.line, dtype=np.float64)
                fine_start = self.project_point(start, coarse_line)
                fine_end = self.project_point(end, coarse_line)
                fine, fine_debug = self._fit_edge_from_image(
                    name,
                    representations,
                    fine_start,
                    fine_end,
                    fine_band,
                    pass_number=2,
                    rough_reference=(start, end),
                )
                if fine.success and fine.score >= coarse.score * 0.82:
                    chosen, chosen_debug = fine, fine_debug
            fitted.append(chosen)
            edge_debug[name] = chosen_debug

        minimum_score = self.config.minimum_edge_fit_score
        weak_indices = [
            index
            for index, result in enumerate(fitted)
            if not result.success or result.score < minimum_score or result.line is None
        ]
        active_lines: list[np.ndarray] = []
        final_edges: list[EdgeFitResult] = []
        for index, result in enumerate(fitted):
            if index in weak_indices:
                rough_line = self._infer_edge_line(
                    index, local_rough, fitted, weak_indices
                )
                final_edges.append(
                    EdgeFitResult(
                        name=EDGE_NAMES[index],
                        success=True,
                        line=tuple(float(value) for value in rough_line),
                        score=min(result.score, 0.34),
                        support_ratio=result.support_ratio,
                        support_length_px=result.support_length_px,
                        continuity=result.continuity,
                        gradient_score=result.gradient_score,
                        residual_px=result.residual_px,
                        orientation_delta_degrees=result.orientation_delta_degrees,
                        rough_offset_px=result.rough_offset_px,
                        signed_rough_offset_px=result.signed_rough_offset_px,
                        candidate_count=result.candidate_count,
                        inlier_count=result.inlier_count,
                        inferred=True,
                    )
                )
                active_lines.append(rough_line)
            else:
                final_edges.append(result)
                active_lines.append(np.asarray(result.line, dtype=np.float64))

        active_lines = self._tune_inferred_edges(
            active_lines, weak_indices, local_rough, short_side
        )
        for index in weak_indices:
            final_edges[index].line = tuple(float(value) for value in active_lines[index])

        excessive_edge = next(
            (
                edge
                for edge in final_edges
                if not edge.inferred
                and edge.rough_offset_px
                > short_side * self.config.maximum_edge_displacement_fraction
            ),
            None,
        )
        if excessive_edge is not None:
            return self._result_with_metrics(
                rough,
                rough,
                final_edges,
                roi_box,
                scale,
                edge_debug,
                roi,
                detector_inferred_edges,
                success=False,
                fallback_reason=(
                    "background_edge_hijack: excessive_edge_displacement "
                    f"({excessive_edge.name})."
                ),
            )

        if len(weak_indices) == 4:
            return self._result_with_metrics(
                rough,
                rough,
                final_edges,
                roi_box,
                scale,
                edge_debug,
                roi,
                detector_inferred_edges,
                success=False,
                fallback_reason="No physical edge had enough distributed support.",
            )

        intersections: list[np.ndarray] = []
        adjacent_pairs = ((3, 0), (0, 1), (1, 2), (2, 3))
        for first, second in adjacent_pairs:
            point = self.intersect_lines(active_lines[first], active_lines[second])
            if point is None:
                return self._result_with_metrics(
                    rough,
                    rough,
                    final_edges,
                    roi_box,
                    scale,
                    edge_debug,
                    roi,
                    detector_inferred_edges,
                    success=False,
                    fallback_reason="Refined neighboring edges do not intersect reliably.",
                )
            intersections.append(point)
        local_refined = np.asarray(intersections, dtype=np.float32)
        local_refined, active_lines, reconstructed_corner_count = (
            self._recover_single_corner_outlier(
                local_rough, local_refined, active_lines, final_edges, short_side
            )
        )
        if reconstructed_corner_count:
            for index, line in enumerate(active_lines):
                final_edges[index].line = tuple(float(value) for value in line)
        valid, fallback_reason = self._validate_refinement(
            local_rough, local_refined, final_edges
        )
        attempted_refined = local_refined.copy()
        attempted_refined[:, 0] = attempted_refined[:, 0] / scale + left
        attempted_refined[:, 1] = attempted_refined[:, 1] / scale + top
        refined = attempted_refined if valid else rough.copy()
        return self._result_with_metrics(
            rough,
            refined,
            final_edges,
            roi_box,
            scale,
            edge_debug,
            roi,
            detector_inferred_edges,
            success=valid,
            reconstructed_corner_count=reconstructed_corner_count,
            diagnostic_refined=attempted_refined,
            fallback_reason=(
                fallback_reason
                if not valid
                else (
                    f"{len(weak_indices)} edge(s) retained rough geometry."
                    if weak_indices
                    else None
                )
            ),
        )

    def _fit_edge_from_image(
        self,
        name: str,
        representations: dict[str, object],
        start: np.ndarray,
        end: np.ndarray,
        band_width: float,
        *,
        pass_number: int,
        rough_reference: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[EdgeFitResult, _EdgeDebug]:
        shape = representations["shape"]
        band_mask, polygon = self.build_search_band(shape, start, end, band_width)  # type: ignore[arg-type]
        expected_line = self.line_from_points(start, end)
        normal = expected_line[:2]
        evidence, alignment = self._directional_evidence(
            representations, normal, band_mask
        )
        active = evidence[band_mask > 0]
        positive = active[active > 0.025]
        threshold = max(
            0.10,
            float(np.percentile(positive, 66)) if positive.size else 1.0,
        )
        candidate_mask = (
            (band_mask > 0)
            & (evidence >= threshold)
            & ((alignment >= 0.48) | (representations["canny"] > 0))  # type: ignore[operator]
        )
        ys, xs = np.nonzero(candidate_mask)
        candidates = np.column_stack((xs, ys)).astype(np.float64)
        if len(candidates):
            candidate_distance = np.abs(
                expected_line[0] * xs
                + expected_line[1] * ys
                + expected_line[2]
            )
            proximity = np.exp(
                -np.square(candidate_distance / max(1.0, band_width * 0.58))
            )
            weights = evidence[ys, xs] * (0.42 + 0.58 * proximity)
            if len(candidates) > self.config.maximum_edge_candidates:
                rng = np.random.default_rng(
                    613 + EDGE_NAMES.index(name) * 109 + pass_number * 17
                )
                probabilities = weights / weights.sum()
                selected = rng.choice(
                    len(candidates),
                    self.config.maximum_edge_candidates,
                    replace=False,
                    p=probabilities,
                )
                candidates = candidates[selected]
                weights = weights[selected]
        else:
            weights = np.empty(0, dtype=np.float64)
        reference_start, reference_end = rough_reference or (start, end)
        result, inlier_mask = self.fit_line_ransac(
            name,
            candidates,
            weights,
            reference_start,
            reference_end,
            band_width,
            distance_threshold=max(
                1.35,
                self._short_side_from_shape(shape)  # type: ignore[arg-type]
                * (0.0028 if pass_number == 1 else 0.0019),
            ),
        )
        return result, _EdgeDebug(
            band_polygon=polygon,
            candidates=candidates,
            inliers=inlier_mask,
            expected_start=np.asarray(reference_start, dtype=np.float32),
            expected_end=np.asarray(reference_end, dtype=np.float32),
            evidence=evidence,
        )

    def fit_line_ransac(
        self,
        name: str,
        points: np.ndarray,
        weights: np.ndarray,
        expected_start: np.ndarray,
        expected_end: np.ndarray,
        band_width: float,
        distance_threshold: float,
    ) -> tuple[EdgeFitResult, np.ndarray]:
        """Fit a direction-constrained line with distributed-support RANSAC."""
        points = np.asarray(points, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        empty_mask = np.zeros(len(points), dtype=bool)
        expected_length = float(np.linalg.norm(expected_end - expected_start))
        if len(points) < 12 or expected_length < 8:
            return EdgeFitResult(name=name, success=False, candidate_count=len(points)), empty_mask
        weights = np.clip(weights, 1e-4, None)
        expected_line = self.line_from_points(expected_start, expected_end)
        tangent = self._line_tangent(expected_line)
        midpoint = (expected_start + expected_end) / 2
        rng = np.random.default_rng(911 + EDGE_NAMES.index(name) * 137)
        probabilities = weights / weights.sum()
        best_objective = -1.0
        best_inliers: np.ndarray | None = None
        for _ in range(self.config.ransac_iterations):
            pair = rng.choice(len(points), 2, replace=False, p=probabilities)
            if abs(float(np.dot(points[pair[1]] - points[pair[0]], tangent))) < expected_length * 0.16:
                continue
            hypothesis = self.line_from_points(points[pair[0]], points[pair[1]])
            hypothesis = self.align_line(hypothesis, expected_line)
            angle_delta = self.line_angle_delta(hypothesis, expected_line)
            if angle_delta > self.config.maximum_edge_angle_delta_degrees:
                continue
            offset = abs(float(np.dot(hypothesis[:2], midpoint) + hypothesis[2]))
            if offset > band_width * 1.12:
                continue
            signed_offset = -float(
                np.dot(hypothesis[:2], midpoint) + hypothesis[2]
            )
            perimeter_score = float(np.exp(-offset / max(1.0, band_width * 0.42)))
            inward_penalty = max(0.0, -signed_offset / max(1.0, band_width))
            distances = np.abs(points @ hypothesis[:2] + hypothesis[2])
            inliers = distances <= distance_threshold
            if int(inliers.sum()) < 10:
                continue
            metrics = self._support_metrics(
                points, weights, inliers, expected_start, expected_end, distances
            )
            objective = (
                0.36 * metrics["support_ratio"]
                + 0.22 * metrics["continuity"]
                + 0.17 * metrics["gradient_score"]
                + 0.11 * metrics["density"]
                + 0.07 * float(weights[inliers].sum() / weights.sum())
                + 0.07 * perimeter_score
                - 0.04 * inward_penalty
            )
            if objective > best_objective:
                best_objective = objective
                best_inliers = inliers
            if best_objective >= 0.91:
                break
        if best_inliers is None:
            return EdgeFitResult(name=name, success=False, candidate_count=len(points)), np.zeros(len(points), dtype=bool)

        line = expected_line.copy()
        inliers = best_inliers
        for _ in range(3):
            line = self._weighted_line(points[inliers], weights[inliers])
            line = self.align_line(line, expected_line)
            distances = np.abs(points @ line[:2] + line[2])
            updated = distances <= distance_threshold
            if int(updated.sum()) < 10 or np.array_equal(updated, inliers):
                break
            inliers = updated
        distances = np.abs(points @ line[:2] + line[2])
        metrics = self._support_metrics(
            points, weights, inliers, expected_start, expected_end, distances
        )
        angle_delta = self.line_angle_delta(line, expected_line)
        offset = abs(float(np.dot(line[:2], midpoint) + line[2]))
        signed_offset = -float(np.dot(line[:2], midpoint) + line[2])
        residual_score = float(
            np.exp(-metrics["residual"] / max(0.5, distance_threshold))
        )
        orientation_score = float(
            np.exp(-angle_delta / max(2.0, self.config.maximum_edge_angle_delta_degrees * 0.55))
        )
        proximity_score = float(np.exp(-offset / max(1.0, band_width * 0.52)))
        inward_penalty = max(0.0, -signed_offset / max(1.0, band_width))
        score = float(np.clip(
            0.23 * metrics["support_ratio"]
            + 0.16 * metrics["continuity"]
            + 0.14 * metrics["gradient_score"]
            + 0.13 * residual_score
            + 0.11 * orientation_score
            + 0.15 * proximity_score
            + 0.08 * metrics["density"]
            - 0.04 * inward_penalty,
            0.0,
            1.0,
        ))
        return EdgeFitResult(
            name=name,
            success=True,
            line=tuple(float(value) for value in line),
            score=score,
            support_ratio=metrics["support_ratio"],
            support_length_px=metrics["support_ratio"] * expected_length,
            continuity=metrics["continuity"],
            gradient_score=metrics["gradient_score"],
            residual_px=metrics["residual"],
            orientation_delta_degrees=angle_delta,
            rough_offset_px=offset,
            signed_rough_offset_px=signed_offset,
            candidate_count=len(points),
            inlier_count=int(inliers.sum()),
        ), inliers

    def build_search_band(
        self,
        shape: tuple[int, int],
        start: np.ndarray,
        end: np.ndarray,
        band_width: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if length < 1:
            return np.zeros(shape, dtype=np.uint8), np.zeros((4, 2), dtype=np.float32)
        tangent = direction / length
        normal = np.asarray((-tangent[1], tangent[0]))
        trim = length * self.config.edge_endpoint_trim_fraction
        first = np.asarray(start, dtype=np.float64) + tangent * trim
        second = np.asarray(end, dtype=np.float64) - tangent * trim
        polygon = np.asarray(
            [
                first + normal * band_width,
                second + normal * band_width,
                second - normal * band_width,
                first - normal * band_width,
            ],
            dtype=np.float32,
        )
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 255)
        return mask, polygon

    @staticmethod
    def line_from_points(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        delta = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
        length = float(np.linalg.norm(delta))
        if length < 1e-8:
            raise ValueError("A line requires two distinct points")
        line = np.asarray(
            (delta[1] / length, -delta[0] / length, 0.0), dtype=np.float64
        )
        line[2] = -float(np.dot(line[:2], first))
        return line

    @staticmethod
    def intersect_lines(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
        matrix = np.asarray((first[:2], second[:2]), dtype=np.float64)
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) < 1e-5:
            return None
        return np.linalg.solve(matrix, -np.asarray((first[2], second[2]))).astype(np.float32)

    @staticmethod
    def project_point(point: np.ndarray, line: np.ndarray) -> np.ndarray:
        distance = float(np.dot(line[:2], point) + line[2])
        return np.asarray(point, dtype=np.float64) - line[:2] * distance

    @staticmethod
    def align_line(line: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return -line if float(np.dot(line[:2], reference[:2])) < 0 else line

    @classmethod
    def line_angle_delta(cls, first: np.ndarray, second: np.ndarray) -> float:
        dot = abs(float(np.dot(cls._line_tangent(first), cls._line_tangent(second))))
        return degrees(acos(float(np.clip(dot, -1.0, 1.0))))

    def _representations(self, rgb: np.ndarray) -> dict[str, object]:
        blurred = cv2.GaussianBlur(rgb, (3, 3), 0.7)
        gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        channels = (
            gray,
            clahe.apply(gray),
            clahe.apply(lab[:, :, 0]),
            lab[:, :, 1],
            lab[:, :, 2],
            clahe.apply(hsv[:, :, 2]),
            hsv[:, :, 1],
        )
        gradients: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        canny = np.zeros_like(gray)
        for channel in channels:
            gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(gx, gy)
            gradients.append((gx, gy, magnitude))
            canny = cv2.bitwise_or(canny, cv2.Canny(channel, 45, 135))
        return {
            "shape": gray.shape,
            "gray": gray,
            "gradients": gradients,
            "canny": canny,
        }

    @staticmethod
    def _directional_evidence(
        representations: dict[str, object],
        normal: np.ndarray,
        band_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        active = band_mask > 0
        strength = np.zeros(band_mask.shape, dtype=np.float32)
        alignment = np.zeros(band_mask.shape, dtype=np.float32)
        for gx, gy, magnitude in representations["gradients"]:  # type: ignore[union-attr]
            directional = np.abs(gx * normal[0] + gy * normal[1])
            values = magnitude[active]
            scale = max(8.0, float(np.percentile(values, 97)) if values.size else 8.0)
            np.maximum(
                strength,
                np.clip(directional / scale, 0.0, 1.0),
                out=strength,
            )
            np.maximum(
                alignment,
                np.clip(directional / (magnitude + 1e-5), 0.0, 1.0),
                out=alignment,
            )
        canny = (representations["canny"] > 0).astype(np.float32)  # type: ignore[operator]
        evidence = np.clip((0.78 * strength + 0.22 * canny) * (0.55 + 0.45 * alignment), 0.0, 1.0)
        return evidence, alignment

    @staticmethod
    def _weighted_line(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        normalized_weights = weights / max(float(weights.sum()), 1e-9)
        center = np.sum(points * normalized_weights[:, None], axis=0)
        centered = points - center
        covariance = (centered * normalized_weights[:, None]).T @ centered
        values, vectors = np.linalg.eigh(covariance)
        tangent = vectors[:, int(np.argmax(values))]
        normal = np.asarray((tangent[1], -tangent[0]), dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1e-9)
        return np.asarray((normal[0], normal[1], -np.dot(normal, center)))

    @classmethod
    def _support_metrics(
        cls,
        points: np.ndarray,
        weights: np.ndarray,
        inliers: np.ndarray,
        expected_start: np.ndarray,
        expected_end: np.ndarray,
        distances: np.ndarray,
    ) -> dict[str, float]:
        length = max(float(np.linalg.norm(expected_end - expected_start)), 1.0)
        tangent = (expected_end - expected_start) / length
        positions = ((points[inliers] - expected_start) @ tangent) / length
        clipped = np.clip(positions, 0.0, 1.0)
        if len(clipped) >= 2:
            low, high = np.percentile(clipped, (3, 97))
            support_ratio = float(np.clip(high - low, 0.0, 1.0))
        else:
            support_ratio = 0.0
        bins = np.unique(np.clip((clipped * 14).astype(int), 0, 13))
        continuity = float(len(bins) / 14)
        gradient_score = float(np.average(weights[inliers]))
        residual = float(np.median(distances[inliers]))
        density = float(np.clip(int(inliers.sum()) / max(12.0, length * 0.85), 0.0, 1.0))
        return {
            "support_ratio": support_ratio,
            "continuity": continuity,
            "gradient_score": gradient_score,
            "residual": residual,
            "density": density,
        }

    def _infer_edge_line(
        self,
        index: int,
        rough: np.ndarray,
        fitted: list[EdgeFitResult],
        weak_indices: list[int],
    ) -> np.ndarray:
        """Conservatively reconstruct an occluded edge from rough geometry.

        The rough edge remains the positional anchor.  A reliable opposite edge
        may contribute a small orientation correction; this avoids replacing a
        hidden physical boundary with a complete printed rectangle.
        """
        rough_line = self.line_from_points(rough[index], rough[(index + 1) % 4])
        opposite = (index + 2) % 4
        if opposite in weak_indices or fitted[opposite].line is None:
            return rough_line
        opposite_line = self.align_line(
            np.asarray(fitted[opposite].line, dtype=np.float64), rough_line
        )
        if self.line_angle_delta(rough_line, opposite_line) > (
            self.config.maximum_edge_angle_delta_degrees * 1.25
        ):
            return rough_line
        normal = rough_line[:2] * 0.82 + opposite_line[:2] * 0.18
        normal /= max(float(np.linalg.norm(normal)), 1e-9)
        midpoint = (rough[index] + rough[(index + 1) % 4]) / 2.0
        return np.asarray((normal[0], normal[1], -np.dot(normal, midpoint)))

    def _tune_inferred_edges(
        self,
        lines: list[np.ndarray],
        weak_indices: list[int],
        rough: np.ndarray,
        short_side: float,
    ) -> list[np.ndarray]:
        """Use the preset ratio only as a tie-breaker near the rough perimeter."""
        if not weak_indices or self.target_ratio is None:
            return lines
        tuned = [line.copy() for line in lines]
        for index in weak_indices:
            baseline = tuned[index].copy()
            best_line = baseline
            best_objective = float("inf")
            for displacement in (-0.008 * short_side, 0.0, 0.008 * short_side):
                candidate = baseline.copy()
                candidate[2] -= displacement
                trial = tuned.copy()
                trial[index] = candidate
                quad = self._intersections_for_lines(trial)
                if quad is None or not cv2.isContourConvex(np.round(quad).astype(np.int32)):
                    continue
                area_ratio = abs(float(cv2.contourArea(quad))) / max(
                    abs(float(cv2.contourArea(rough))), 1.0
                )
                ratio_error = abs(log(self._quad_ratio(quad) / self.target_ratio))
                center_shift = float(np.linalg.norm(quad.mean(axis=0) - rough.mean(axis=0)))
                objective = (
                    0.58 * abs(area_ratio - 1.0)
                    + 0.24 * ratio_error
                    + 0.18 * center_shift / max(short_side, 1.0)
                )
                if objective < best_objective:
                    best_objective = objective
                    best_line = candidate
            tuned[index] = best_line
        return tuned

    def _intersections_for_lines(
        self, lines: list[np.ndarray]
    ) -> np.ndarray | None:
        points: list[np.ndarray] = []
        for first, second in ((3, 0), (0, 1), (1, 2), (2, 3)):
            point = self.intersect_lines(lines[first], lines[second])
            if point is None:
                return None
            points.append(point)
        return np.asarray(points, dtype=np.float32)

    def _recover_single_corner_outlier(
        self,
        rough: np.ndarray,
        refined: np.ndarray,
        lines: list[np.ndarray],
        edges: list[EdgeFitResult],
        short_side: float,
    ) -> tuple[np.ndarray, list[np.ndarray], int]:
        """Repair one inward corner by anchoring its weaker edge at the stable end.

        This is deliberately local: three mutually consistent corners are kept,
        and the questionable adjacent edge is rebuilt from rough-perimeter
        geometry.  It therefore cannot turn a local finger/corner defect into a
        smaller internal rectangle.
        """
        distances = np.linalg.norm(refined - rough, axis=1)
        ordered = np.argsort(distances)
        corner = int(ordered[-1])
        others = distances[ordered[:-1]]
        threshold = max(
            short_side * 0.052,
            float(np.median(others)) * 1.85 + short_side * 0.008,
        )
        inward_vector = rough.mean(axis=0) - rough[corner]
        movement = refined[corner] - rough[corner]
        if distances[corner] <= threshold or float(np.dot(movement, inward_vector)) <= 0:
            return refined, lines, 0

        adjacent = ((corner - 1) % 4, corner)
        suspect = min(adjacent, key=lambda index: edges[index].score)
        if (
            not edges[suspect].inferred
            and edges[suspect].score >= self.config.strong_outer_boundary_score
        ):
            return refined, lines, 0

        other_corner = (corner - 1) % 4 if suspect == (corner - 1) % 4 else (corner + 1) % 4
        rough_line = self.line_from_points(rough[suspect], rough[(suspect + 1) % 4])
        anchor = refined[other_corner]
        rebuilt = rough_line.copy()
        rebuilt[2] = -float(np.dot(rebuilt[:2], anchor))
        trial_lines = [line.copy() for line in lines]
        trial_lines[suspect] = rebuilt
        trial = self._intersections_for_lines(trial_lines)
        if trial is None or not cv2.isContourConvex(np.round(trial).astype(np.int32)):
            return refined, lines, 0
        trial_distances = np.linalg.norm(trial - rough, axis=1)
        rough_area = max(abs(float(cv2.contourArea(rough))), 1.0)
        original_area_error = abs(abs(float(cv2.contourArea(refined))) / rough_area - 1.0)
        trial_area_error = abs(abs(float(cv2.contourArea(trial))) / rough_area - 1.0)
        if (
            trial_distances[corner] < distances[corner] * 0.82
            and trial_area_error <= original_area_error + 0.025
        ):
            edges[suspect].inferred = True
            edges[suspect].score = min(edges[suspect].score, 0.40)
            return trial, trial_lines, 1
        return refined, lines, 0

    @staticmethod
    def _quad_dimensions(points: np.ndarray) -> tuple[float, float]:
        width = 0.5 * (
            float(np.linalg.norm(points[1] - points[0]))
            + float(np.linalg.norm(points[2] - points[3]))
        )
        height = 0.5 * (
            float(np.linalg.norm(points[2] - points[1]))
            + float(np.linalg.norm(points[3] - points[0]))
        )
        return width, height

    def _signed_edge_displacements(
        self, rough: np.ndarray, refined: np.ndarray
    ) -> tuple[float, float, float, float]:
        values: list[float] = []
        for index in range(4):
            rough_line = self.line_from_points(
                rough[index], rough[(index + 1) % 4]
            )
            midpoint = (refined[index] + refined[(index + 1) % 4]) / 2.0
            values.append(float(np.dot(rough_line[:2], midpoint) + rough_line[2]))
        return tuple(values)  # type: ignore[return-value]

    def _validate_refinement(
        self,
        rough: np.ndarray,
        refined: np.ndarray,
        edges: list[EdgeFitResult] | None = None,
    ) -> tuple[bool, str | None]:
        if not np.isfinite(refined).all():
            return False, "Refined line intersections are not finite."
        contour = refined.astype(np.float32)
        if not cv2.isContourConvex(np.round(contour).astype(np.int32)):
            return False, "Refined corners do not form a convex card."
        rough_area = abs(float(cv2.contourArea(rough)))
        refined_area = abs(float(cv2.contourArea(refined)))
        area_ratio = refined_area / max(rough_area, 1.0)
        if rough_area < 1 or area_ratio < self.config.minimum_refined_area_fraction:
            return False, "Refined card area disagrees with detection geometry."
        if area_ratio > self.config.maximum_refined_area_expansion:
            return False, "background_edge_hijack: excessive_area_expansion."
        rough_width, rough_height = self._quad_dimensions(rough)
        refined_width, refined_height = self._quad_dimensions(refined)
        width_ratio = refined_width / max(rough_width, 1.0)
        height_ratio = refined_height / max(rough_height, 1.0)
        if width_ratio < self.config.minimum_refined_width_fraction:
            return False, "width_collapse: refined width moved inside the rough perimeter."
        if height_ratio < self.config.minimum_refined_height_fraction:
            return False, "height_collapse: refined height moved inside the rough perimeter."
        if width_ratio > self.config.maximum_refined_dimension_expansion:
            return False, "background_edge_hijack: excessive_width_expansion."
        if height_ratio > self.config.maximum_refined_dimension_expansion:
            return False, "background_edge_hijack: excessive_height_expansion."
        short_side = self._short_side(rough)
        distances = np.linalg.norm(refined - rough, axis=1)
        if float(distances.max()) > short_side * self.config.maximum_corner_displacement_fraction:
            return False, "Refined corners moved too far from detection geometry."
        rough_center = rough.mean(axis=0)
        refined_center = refined.mean(axis=0)
        center_displacement = float(np.linalg.norm(refined_center - rough_center))
        if center_displacement > short_side * self.config.maximum_center_displacement_fraction:
            return False, "Refined card center moved too far from detection geometry."
        signed_edge_displacements = self._signed_edge_displacements(rough, refined)
        if max(abs(value) for value in signed_edge_displacements) > (
            short_side * self.config.maximum_edge_displacement_fraction
        ):
            return False, "Refined card edge moved too far from detection geometry."
        maximum_inward = max(0.0, -min(signed_edge_displacements))
        strong_outer_evidence = bool(
            edges
            and sum(
                not edge.inferred
                and edge.score >= self.config.strong_outer_boundary_score
                and edge.support_ratio >= self.config.strong_outer_boundary_support
                for edge in edges
            ) >= 3
            and all(
                edge.inferred
                or edge.score >= self.config.strong_outer_boundary_score
                for edge in edges
                if edge.signed_rough_offset_px < 0
            )
        )
        if (
            area_ratio < self.config.collapse_area_fraction
            and maximum_inward
            > short_side * self.config.collapse_inward_displacement_fraction
            and not strong_outer_evidence
        ):
            return False, "quadrilateral_collapse: inward refinement lacks strong outer-boundary evidence."
        edges = [float(np.linalg.norm(refined[(i + 1) % 4] - refined[i])) for i in range(4)]
        if min(edges) < max(6.0, short_side * 0.18):
            return False, "A refined card edge collapsed."
        if self.target_ratio is not None:
            rough_error = abs(log(self._quad_ratio(rough) / self.target_ratio))
            refined_error = abs(log(self._quad_ratio(refined) / self.target_ratio))
            error_increase = refined_error - rough_error
            if error_increase > self.config.maximum_ratio_error_increase:
                return False, "background_edge_hijack: ratio became less card-like."
            if (
                area_ratio > self.config.hijack_area_expansion
                and error_increase > self.config.hijack_ratio_error_increase
            ):
                return False, "background_edge_hijack: outward expansion worsened ratio."
        return True, None

    def _result_with_metrics(
        self,
        rough: np.ndarray,
        refined: np.ndarray,
        edges: list[EdgeFitResult],
        roi_box: tuple[int, int, int, int],
        scale: float,
        edge_debug: dict[str, _EdgeDebug],
        roi: np.ndarray,
        detector_inferred_edges: int,
        *,
        success: bool,
        fallback_reason: str | None,
        diagnostic_refined: np.ndarray | None = None,
        reconstructed_corner_count: int = 0,
    ) -> CornerRefinementResult:
        edge_scores = [edge.score for edge in edges]
        corner_confidences = tuple(
            min(edge_scores[first], edge_scores[second])
            for first, second in ((3, 0), (0, 1), (1, 2), (2, 3))
        )
        diagnostic = refined if diagnostic_refined is None else diagnostic_refined
        displacements = np.linalg.norm(diagnostic - rough, axis=1)
        rough_area = max(abs(float(cv2.contourArea(rough))), 1.0)
        refined_area_ratio = abs(float(cv2.contourArea(diagnostic))) / rough_area
        refined_area = abs(float(cv2.contourArea(diagnostic)))
        rough_width, rough_height = self._quad_dimensions(rough)
        refined_width, refined_height = self._quad_dimensions(diagnostic)
        rough_ratio = self._quad_ratio(rough)
        refined_ratio = self._quad_ratio(diagnostic)
        short_side = max(self._short_side(rough), 1.0)
        agreement = float(np.exp(-float(displacements.max()) / (short_side * 0.07)))
        strong_count = sum(not edge.inferred for edge in edges)
        confidence = float(np.clip(
            0.42 * float(np.mean(edge_scores))
            + 0.18 * min(edge_scores)
            + 0.22 * agreement
            + 0.18 * (strong_count / 4)
            - min(0.18, detector_inferred_edges * 0.07),
            0.0,
            1.0,
        ))
        if strong_count <= 2:
            confidence = min(confidence, 0.48)
        if not success:
            confidence = min(confidence, 0.34)

        converted_edges: list[EdgeFitResult] = []
        left, top, _, _ = roi_box
        for edge in edges:
            line = None
            if edge.line is not None:
                local = np.asarray(edge.line, dtype=np.float64)
                line = (
                    float(local[0]),
                    float(local[1]),
                    float(local[2] / scale - local[0] * left - local[1] * top),
                )
            converted_edges.append(
                EdgeFitResult(
                    name=edge.name,
                    success=edge.success,
                    line=line,
                    score=edge.score,
                    support_ratio=edge.support_ratio,
                    support_length_px=edge.support_length_px / scale,
                    continuity=edge.continuity,
                    gradient_score=edge.gradient_score,
                    residual_px=edge.residual_px / scale,
                    orientation_delta_degrees=edge.orientation_delta_degrees,
                    rough_offset_px=edge.rough_offset_px / scale,
                    signed_rough_offset_px=edge.signed_rough_offset_px / scale,
                    candidate_count=edge.candidate_count,
                    inlier_count=edge.inlier_count,
                    inferred=edge.inferred,
                )
            )
        metrics: dict[str, object] = {
            "edge_scores": {
                edge.name: round(edge.score, 4) for edge in converted_edges
            },
            "edge_support_lengths": {
                edge.name: round(edge.support_length_px, 2)
                for edge in converted_edges
            },
            "edge_residuals": {
                edge.name: round(edge.residual_px, 3) for edge in converted_edges
            },
            "edge_confidences": {
                edge.name: round(edge.score, 4) for edge in converted_edges
            },
            "rough_to_refined_corner_distances": tuple(
                round(float(value), 3) for value in displacements
            ),
            "rough_to_refined_edge_angle_deltas": {
                edge.name: round(edge.orientation_delta_degrees, 3)
                for edge in converted_edges
            },
            "refinement_confidence": round(confidence, 4),
            "fallback_reason": fallback_reason,
            "roi_box": roi_box,
            "roi_scale": round(scale, 5),
            "strong_edge_count": strong_count,
            "refined_area_ratio": round(refined_area_ratio, 5),
            "rough_area": round(rough_area, 3),
            "refined_area": round(refined_area, 3),
            "area_ratio": round(refined_area_ratio, 5),
            "rough_width": round(rough_width, 3),
            "refined_width": round(refined_width, 3),
            "width_ratio": round(refined_width / max(rough_width, 1.0), 5),
            "rough_height": round(rough_height, 3),
            "refined_height": round(refined_height, 3),
            "height_ratio": round(refined_height / max(rough_height, 1.0), 5),
            "rough_center": tuple(round(float(value), 3) for value in rough.mean(axis=0)),
            "refined_center": tuple(round(float(value), 3) for value in diagnostic.mean(axis=0)),
            "corner_displacements_px": {
                name: round(float(value), 3)
                for name, value in zip(("top_left", "top_right", "bottom_right", "bottom_left"), displacements)
            },
            "reconstructed_corner_count": reconstructed_corner_count,
            "center_displacement_px": round(
                float(np.linalg.norm(diagnostic.mean(axis=0) - rough.mean(axis=0))), 3
            ),
            "signed_edge_displacements_px": {
                name: round(value, 3)
                for name, value in zip(
                    EDGE_NAMES, self._signed_edge_displacements(rough, diagnostic)
                )
            },
            "rough_geometry_ratio": round(rough_ratio, 5),
            "refined_geometry_ratio": round(refined_ratio, 5),
            "target_ratio": (
                round(self.target_ratio, 5) if self.target_ratio is not None else None
            ),
            "background_hijack_detected": bool(
                fallback_reason and "background_edge_hijack" in fallback_reason
            ),
            "refinement_rejection_reason": fallback_reason if not success else None,
        }
        if self.debug:
            metrics["stage_images"] = self._debug_images(
                roi, rough, diagnostic, roi_box, scale, edge_debug, edges, success
            )
        LOGGER.debug(
            "corner_refinement %s",
            {
                "edge_top_score": round(edge_scores[0], 4),
                "edge_right_score": round(edge_scores[1], 4),
                "edge_bottom_score": round(edge_scores[2], 4),
                "edge_left_score": round(edge_scores[3], 4),
                "edge_support_lengths": metrics["edge_support_lengths"],
                "rough_to_refined_corner_distances": metrics[
                    "rough_to_refined_corner_distances"
                ],
                "refinement_confidence": round(confidence, 4),
                "fallback_reason": fallback_reason,
            },
        )
        return CornerRefinementResult(
            success=success,
            rough_corners=self._point_tuple(rough),
            refined_corners=self._point_tuple(refined),
            edge_results=tuple(converted_edges),
            edge_confidences=tuple(edge.score for edge in converted_edges),
            corner_confidences=corner_confidences,
            confidence=confidence,
            reconstructed_corner_count=reconstructed_corner_count,
            roi_box=roi_box,
            fallback_reason=fallback_reason,
            debug_info=metrics,
        )

    def _debug_images(
        self,
        roi: np.ndarray,
        rough: np.ndarray,
        refined: np.ndarray,
        roi_box: tuple[int, int, int, int],
        scale: float,
        debug: dict[str, _EdgeDebug],
        edges: list[EdgeFitResult],
        success: bool,
    ) -> dict[str, Image.Image]:
        left, top, _, _ = roi_box
        rough_local = rough.copy()
        refined_local = refined.copy()
        for points in (rough_local, refined_local):
            points[:, 0] = (points[:, 0] - left) * scale
            points[:, 1] = (points[:, 1] - top) * scale
        bands = roi.copy()
        fit = roi.copy()
        evidence = np.zeros(roi.shape[:2], dtype=np.float32)
        colors = ((245, 80, 75), (70, 145, 245), (245, 190, 45), (170, 80, 235))
        for index, name in enumerate(EDGE_NAMES):
            item = debug[name]
            overlay = bands.copy()
            cv2.fillConvexPoly(
                overlay, np.round(item.band_polygon).astype(np.int32), colors[index]
            )
            bands = cv2.addWeighted(bands, 0.86, overlay, 0.14, 0)
            evidence = np.maximum(evidence, item.evidence)
            if len(item.candidates):
                rejected = item.candidates[~item.inliers]
                accepted = item.candidates[item.inliers]
                for point in rejected[:: max(1, len(rejected) // 700 + 1)]:
                    cv2.circle(fit, tuple(np.round(point).astype(int)), 1, (235, 65, 165), -1)
                for point in accepted[:: max(1, len(accepted) // 900 + 1)]:
                    cv2.circle(fit, tuple(np.round(point).astype(int)), 1, (40, 225, 210), -1)
            if edges[index].line is not None:
                line = np.asarray(edges[index].line, dtype=np.float64)
                first = self.project_point(item.expected_start, line)
                second = self.project_point(item.expected_end, line)
                direction = second - first
                cv2.line(
                    fit,
                    tuple(np.round(first - direction * 0.08).astype(int)),
                    tuple(np.round(second + direction * 0.08).astype(int)),
                    colors[index],
                    2,
                    cv2.LINE_AA,
                )
        cv2.polylines(
            bands, [np.round(rough_local).astype(np.int32)], True, (255, 165, 35), 3
        )
        cv2.polylines(
            fit, [np.round(rough_local).astype(np.int32)], True, (255, 165, 35), 2
        )
        refined_color = (40, 225, 95) if success else (235, 65, 80)
        cv2.polylines(
            fit, [np.round(refined_local).astype(np.int32)], True, refined_color, 3
        )
        for point in refined_local:
            cv2.circle(fit, tuple(np.round(point).astype(int)), 7, refined_color, 2)
        evidence_image = np.clip(evidence * 255, 0, 255).astype(np.uint8)
        return {
            "rough_and_search_bands": Image.fromarray(bands).copy(),
            "raw_edge_evidence": Image.fromarray(evidence_image).copy(),
            "fitted_edges_and_intersections": Image.fromarray(fit).copy(),
        }

    def _roi_box(
        self, points: np.ndarray, image_width: int, image_height: int
    ) -> tuple[int, int, int, int]:
        width = max(
            float(np.linalg.norm(points[1] - points[0])),
            float(np.linalg.norm(points[2] - points[3])),
        )
        height = max(
            float(np.linalg.norm(points[2] - points[1])),
            float(np.linalg.norm(points[3] - points[0])),
        )
        padding = min(width, height) * self.config.roi_padding_fraction
        return (
            max(0, int(np.floor(points[:, 0].min() - padding))),
            max(0, int(np.floor(points[:, 1].min() - padding))),
            min(image_width, int(np.ceil(points[:, 0].max() + padding)) + 1),
            min(image_height, int(np.ceil(points[:, 1].max() + padding)) + 1),
        )

    @staticmethod
    def _quad_ratio(points: np.ndarray) -> float:
        lengths = [
            float(np.linalg.norm(points[(index + 1) % 4] - points[index]))
            for index in range(4)
        ]
        first = (lengths[0] + lengths[2]) / 2.0
        second = (lengths[1] + lengths[3]) / 2.0
        return max(first, second) / max(1e-6, min(first, second))

    @staticmethod
    def _short_side(points: np.ndarray) -> float:
        return min(
            float(np.linalg.norm(points[(index + 1) % 4] - points[index]))
            for index in range(4)
        )

    @staticmethod
    def _short_side_from_shape(shape: tuple[int, int]) -> float:
        return float(min(shape))

    @staticmethod
    def _line_tangent(line: np.ndarray) -> np.ndarray:
        return np.asarray((-line[1], line[0]), dtype=np.float64)

    @staticmethod
    def _point_tuple(points: np.ndarray) -> tuple[Point, ...]:
        return tuple((float(point[0]), float(point[1])) for point in points)

    @staticmethod
    def _fallback(
        rough: tuple[Point, ...],
        reason: str,
        roi_box: tuple[int, int, int, int] | None = None,
    ) -> CornerRefinementResult:
        return CornerRefinementResult(
            success=False,
            rough_corners=rough,
            refined_corners=rough,
            confidence=0.0,
            roi_box=roi_box,
            fallback_reason=reason,
            debug_info={"fallback_reason": reason},
        )
