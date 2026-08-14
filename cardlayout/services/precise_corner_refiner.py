from __future__ import annotations

import logging
from dataclasses import dataclass
from math import acos, degrees

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
    ) -> None:
        self.config = config or PerspectiveConfig()
        self.debug = debug

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
                rough_line = self.line_from_points(
                    local_rough[index], local_rough[(index + 1) % 4]
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
                        candidate_count=result.candidate_count,
                        inlier_count=result.inlier_count,
                        inferred=True,
                    )
                )
                active_lines.append(rough_line)
            else:
                final_edges.append(result)
                active_lines.append(np.asarray(result.line, dtype=np.float64))

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
        valid, fallback_reason = self._validate_refinement(local_rough, local_refined)
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
            distances = np.abs(points @ hypothesis[:2] + hypothesis[2])
            inliers = distances <= distance_threshold
            if int(inliers.sum()) < 10:
                continue
            metrics = self._support_metrics(
                points, weights, inliers, expected_start, expected_end, distances
            )
            objective = (
                0.38 * metrics["support_ratio"]
                + 0.24 * metrics["continuity"]
                + 0.18 * metrics["gradient_score"]
                + 0.12 * metrics["density"]
                + 0.08 * float(weights[inliers].sum() / weights.sum())
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
        residual_score = float(
            np.exp(-metrics["residual"] / max(0.5, distance_threshold))
        )
        orientation_score = float(
            np.exp(-angle_delta / max(2.0, self.config.maximum_edge_angle_delta_degrees * 0.55))
        )
        proximity_score = float(np.exp(-offset / max(1.0, band_width * 0.52)))
        score = float(np.clip(
            0.25 * metrics["support_ratio"]
            + 0.17 * metrics["continuity"]
            + 0.15 * metrics["gradient_score"]
            + 0.14 * residual_score
            + 0.12 * orientation_score
            + 0.09 * proximity_score
            + 0.08 * metrics["density"],
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

    def _validate_refinement(
        self, rough: np.ndarray, refined: np.ndarray
    ) -> tuple[bool, str | None]:
        if not np.isfinite(refined).all():
            return False, "Refined line intersections are not finite."
        contour = refined.astype(np.float32)
        if not cv2.isContourConvex(np.round(contour).astype(np.int32)):
            return False, "Refined corners do not form a convex card."
        rough_area = abs(float(cv2.contourArea(rough)))
        refined_area = abs(float(cv2.contourArea(refined)))
        if rough_area < 1 or not 0.72 <= refined_area / rough_area <= 1.30:
            return False, "Refined card area disagrees with detection geometry."
        short_side = self._short_side(rough)
        distances = np.linalg.norm(refined - rough, axis=1)
        if float(distances.max()) > short_side * self.config.maximum_corner_displacement_fraction:
            return False, "Refined corners moved too far from detection geometry."
        edges = [float(np.linalg.norm(refined[(i + 1) % 4] - refined[i])) for i in range(4)]
        if min(edges) < max(6.0, short_side * 0.18):
            return False, "A refined card edge collapsed."
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
    ) -> CornerRefinementResult:
        edge_scores = [edge.score for edge in edges]
        corner_confidences = tuple(
            min(edge_scores[first], edge_scores[second])
            for first, second in ((3, 0), (0, 1), (1, 2), (2, 3))
        )
        diagnostic = refined if diagnostic_refined is None else diagnostic_refined
        displacements = np.linalg.norm(diagnostic - rough, axis=1)
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
        }
        if self.debug:
            metrics["stage_images"] = self._debug_images(
                roi, rough, diagnostic, roi_box, scale, edge_debug, edges
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
            corner_confidences=corner_confidences,
            confidence=confidence,
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
        cv2.polylines(
            fit, [np.round(refined_local).astype(np.int32)], True, (40, 225, 95), 3
        )
        for point in refined_local:
            cv2.circle(fit, tuple(np.round(point).astype(int)), 7, (40, 225, 95), 2)
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
        padding_x = width * self.config.roi_padding_fraction
        padding_y = height * self.config.roi_padding_fraction
        return (
            max(0, int(np.floor(points[:, 0].min() - padding_x))),
            max(0, int(np.floor(points[:, 1].min() - padding_y))),
            min(image_width, int(np.ceil(points[:, 0].max() + padding_x)) + 1),
            min(image_height, int(np.ceil(points[:, 1].max() + padding_y)) + 1),
        )

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
