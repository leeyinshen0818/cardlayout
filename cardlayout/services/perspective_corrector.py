from __future__ import annotations

from math import acos, degrees
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from cardlayout.models.card_size import CardSizePreset
from cardlayout.models.detection import Point
from cardlayout.models.perspective import PerspectiveConfig, PerspectiveResult
from cardlayout.services.precise_corner_refiner import PreciseCornerRefiner


class PerspectiveGeometryError(ValueError):
    """Raised when four points cannot describe a usable card plane."""


class PerspectiveCorrector:
    """Rectify original-resolution card geometry with a single homography."""

    def __init__(
        self,
        card_size: CardSizePreset,
        config: PerspectiveConfig | None = None,
        debug: bool = False,
    ) -> None:
        self.card_size = card_size
        self.config = config or PerspectiveConfig()
        self.debug = debug
        self.corner_refiner = PreciseCornerRefiner(
            self.config, debug=debug, target_ratio=self.target_ratio
        )

    @property
    def target_ratio(self) -> float:
        return max(
            self.card_size.width_mm / self.card_size.height_mm,
            self.card_size.height_mm / self.card_size.width_mm,
        )

    def correct(
        self,
        image: Image.Image,
        points: Iterable[Point],
        *,
        detector_confidence: float = 1.0,
        inferred_corner_count: int = 0,
        method: str = "automatic",
        refine: bool = True,
        pdf_frame_background: tuple[int, int, int] | None = None,
    ) -> PerspectiveResult:
        """Order, validate, optionally refine, and rectify four source points."""
        source = tuple((float(x), float(y)) for x, y in points)
        perspective_method = "manual" if method == "manual" else "automatic"
        try:
            ordered = self.order_corners(source)
        except PerspectiveGeometryError as exc:
            return PerspectiveResult(
                success=False,
                source_points=source,
                method=perspective_method,
                warning=str(exc),
            )

        rgb = image.convert("RGB")
        working = np.asarray(rgb)
        height, width = working.shape[:2]
        clamped, outside_fraction = self._clamp_to_image(ordered, width, height)
        valid, warning = self.validate_quad(clamped, (width, height))
        if not valid:
            return PerspectiveResult(
                success=False,
                source_points=self._point_tuple(clamped),
                refined_points=self._point_tuple(clamped),
                method=perspective_method,
                warning=warning,
            )

        refined = clamped.copy()
        refinement_shift = 0.0
        refinement_confidence = 1.0 if perspective_method == "manual" else 0.0
        refinement_accepted = perspective_method == "manual" or not refine
        refinement_fallback_reason: str | None = None
        reconstructed_corner_count = 0
        corner_confidences: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
        edge_results = ()
        edge_confidences: tuple[float, ...] = ()
        refinement_debug: dict[str, object] = {}
        if refine and perspective_method == "automatic":
            refinement = self.corner_refiner.refine(
                working,
                clamped,
                detector_inferred_edges=inferred_corner_count,
            )
            refinement_confidence = refinement.confidence
            refinement_accepted = refinement.success
            refinement_fallback_reason = refinement.fallback_reason
            reconstructed_corner_count = refinement.reconstructed_corner_count
            corner_confidences = refinement.corner_confidences or (
                0.0,
                0.0,
                0.0,
                0.0,
            )
            edge_results = refinement.edge_results
            edge_confidences = refinement.edge_confidences
            refinement_debug = refinement.debug_info
            if refinement.success:
                refined = np.asarray(refinement.refined_corners, dtype=np.float32)
            refinement_shift = max(
                (
                    float(np.linalg.norm(refined[index] - clamped[index]))
                    for index in range(4)
                ),
                default=0.0,
            )

        output_width, output_height = self._output_size(refined)
        destination = np.asarray(
            [
                (0.0, 0.0),
                (float(output_width - 1), 0.0),
                (float(output_width - 1), float(output_height - 1)),
                (0.0, float(output_height - 1)),
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(refined.astype(np.float32), destination)
        if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
            return PerspectiveResult(
                success=False,
                source_points=self._point_tuple(clamped),
                refined_points=self._point_tuple(refined),
                destination_points=self._point_tuple(destination),
                method=perspective_method,
                warning="The selected corners produce an unstable correction.",
            )

        rectified = cv2.warpPerspective(
            working,
            matrix,
            (output_width, output_height),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        )
        top_frame = self._residual_top_frame_metrics(
            rectified, pdf_frame_background
        )
        selected_top_edge_offset = 0.0
        if (
            perspective_method == "automatic"
            and pdf_frame_background is not None
            and top_frame["strip_rows"] >= 2
        ):
            adjustment_fraction = min(
                self.config.maximum_residual_top_adjustment_fraction,
                max(0.0, top_frame["strip_rows"] / output_height),
            )
            adjusted = refined.copy()
            adjusted[0] += (refined[3] - refined[0]) * adjustment_fraction
            adjusted[1] += (refined[2] - refined[1]) * adjustment_fraction
            candidate_valid, _ = self.validate_quad(adjusted, (width, height))
            if candidate_valid:
                candidate_width, candidate_height = self._output_size(adjusted)
                candidate_destination = np.asarray(
                    (
                        (0.0, 0.0),
                        (float(candidate_width - 1), 0.0),
                        (float(candidate_width - 1), float(candidate_height - 1)),
                        (0.0, float(candidate_height - 1)),
                    ),
                    dtype=np.float32,
                )
                candidate_matrix = cv2.getPerspectiveTransform(
                    adjusted.astype(np.float32), candidate_destination
                )
                candidate_rectified = cv2.warpPerspective(
                    working,
                    candidate_matrix,
                    (candidate_width, candidate_height),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                candidate_top = self._residual_top_frame_metrics(
                    candidate_rectified, pdf_frame_background
                )
                if (
                    candidate_top["top_white_ratio"]
                    <= top_frame["top_white_ratio"] - 0.18
                ):
                    selected_top_edge_offset = float(
                        0.5
                        * (
                            np.linalg.norm(adjusted[0] - refined[0])
                            + np.linalg.norm(adjusted[1] - refined[1])
                        )
                    )
                    refined = adjusted
                    output_width, output_height = candidate_width, candidate_height
                    destination = candidate_destination
                    matrix = candidate_matrix
                    rectified = candidate_rectified
                    top_frame = candidate_top
        confidence, confidence_level, status, quality = self._quality(
            refined,
            (width, height),
            detector_confidence,
            inferred_corner_count,
            outside_fraction,
            perspective_method,
            refinement_confidence,
            refinement_accepted,
        )
        messages: list[str] = []
        if outside_fraction > self.config.boundary_tolerance_fraction:
            messages.append("Card may be partially outside the photo.")
        if inferred_corner_count:
            messages.append("Some card geometry was reconstructed.")
        if refinement_fallback_reason and status == "review":
            messages.append("Automatic edge refinement needs review.")
        if status == "review" and not messages:
            messages.append("Please review the corrected corners.")

        matrix_tuple = tuple(
            tuple(float(value) for value in row) for row in matrix
        )
        debug_info = {
            "source_corner_points": self._point_tuple(clamped),
            "refined_corner_points": self._point_tuple(refined),
            "inferred_corner_count": inferred_corner_count,
            "reconstructed_corner_count": reconstructed_corner_count,
            "destination_rectangle": self._point_tuple(destination),
            "perspective_confidence": round(confidence, 4),
            "output_dimensions": (output_width, output_height),
            "refinement_shift_px": round(refinement_shift, 3),
            "refinement_confidence": round(refinement_confidence, 4),
            "corner_confidences": tuple(
                round(value, 4) for value in corner_confidences
            ),
            "edge_confidences": {
                edge.name: round(edge.score, 4) for edge in edge_results
            },
            "refinement_fallback_reason": refinement_fallback_reason,
            "post_rectify_top_white_ratio": round(
                top_frame["top_white_ratio"], 4
            ),
            "post_rectify_top_strip_rows": int(top_frame["strip_rows"]),
            "selected_top_edge_offset": round(selected_top_edge_offset, 3),
            "top_candidate_outer_score": round(
                top_frame["outer_score"], 4
            ),
            "top_candidate_inner_score": round(
                top_frame["inner_score"], 4
            ),
            "pdf_frame_edge_penalty": round(
                top_frame["frame_penalty"], 4
            ),
            "refinement": {
                key: value
                for key, value in refinement_debug.items()
                if key != "stage_images"
            },
            "quality": quality,
        }
        if self.debug:
            debug_info["transform_matrix"] = matrix_tuple
            if "stage_images" in refinement_debug:
                debug_info["stage_images"] = refinement_debug["stage_images"]
            debug_info.setdefault("stage_images", {})["final_rectified_result"] = (
                Image.fromarray(rectified).copy()
            )

        return PerspectiveResult(
            success=True,
            source_points=self._point_tuple(clamped),
            refined_points=self._point_tuple(refined),
            destination_points=self._point_tuple(destination),
            rectified_image=Image.fromarray(rectified).copy(),
            confidence=confidence,
            confidence_level=confidence_level,
            status=status,
            method=perspective_method,
            warning=" ".join(messages) or None,
            inferred_corner_count=inferred_corner_count,
            reconstructed_corner_count=reconstructed_corner_count,
            output_dimensions=(output_width, output_height),
            transform_matrix=matrix_tuple,
            corner_confidences=corner_confidences,
            edge_confidences=edge_confidences,
            refinement_confidence=refinement_confidence,
            refinement_fallback_reason=refinement_fallback_reason,
            edge_results=edge_results,
            debug_info=debug_info,
        )

    @staticmethod
    def order_corners(points: Iterable[Point]) -> np.ndarray:
        """Return arbitrary four-corner input as TL, TR, BR, BL."""
        array = np.asarray(tuple(points), dtype=np.float32)
        if array.shape != (4, 2) or not np.isfinite(array).all():
            raise PerspectiveGeometryError("Exactly four finite corners are required.")
        if min(
            float(np.linalg.norm(array[i] - array[j]))
            for i in range(4)
            for j in range(i + 1, 4)
        ) < 1.0:
            raise PerspectiveGeometryError("Two or more corners overlap.")

        hull = cv2.convexHull(array).reshape(-1, 2)
        if len(hull) != 4:
            raise PerspectiveGeometryError("Corners must form a convex four-sided shape.")
        center = hull.mean(axis=0)
        angles = np.arctan2(hull[:, 1] - center[1], hull[:, 0] - center[0])
        cycle = hull[np.argsort(angles)]
        edge_lengths = np.asarray(
            [np.linalg.norm(cycle[(i + 1) % 4] - cycle[i]) for i in range(4)]
        )
        pair_start = 0 if edge_lengths[0] + edge_lengths[2] >= edge_lengths[1] + edge_lengths[3] else 1
        width_edges = (pair_start, (pair_start + 2) % 4)
        midpoints = [
            (cycle[index] + cycle[(index + 1) % 4]) / 2 for index in width_edges
        ]
        delta = midpoints[0] - midpoints[1]
        compare_axis = 1 if abs(float(delta[1])) >= abs(float(delta[0])) else 0
        top_index = width_edges[
            0 if midpoints[0][compare_axis] <= midpoints[1][compare_axis] else 1
        ]
        top = [cycle[top_index], cycle[(top_index + 1) % 4]]
        bottom = [cycle[(top_index + 2) % 4], cycle[(top_index + 3) % 4]]
        top_delta = top[1] - top[0]
        direction_axis = 0 if abs(float(top_delta[0])) >= abs(float(top_delta[1])) else 1
        if top[0][direction_axis] <= top[1][direction_axis]:
            top_left, top_right = top
        else:
            top_left, top_right = top[1], top[0]
        width_direction = top_right - top_left
        width_direction /= max(float(np.linalg.norm(width_direction)), 1e-6)
        bottom.sort(key=lambda point: float(np.dot(point - top_left, width_direction)))
        bottom_left, bottom_right = bottom
        ordered = np.asarray(
            [top_left, top_right, bottom_right, bottom_left], dtype=np.float32
        )
        if not cv2.isContourConvex(ordered.astype(np.int32)):
            ordered[[2, 3]] = ordered[[3, 2]]
        return ordered

    def validate_quad(
        self,
        points: Iterable[Point] | np.ndarray,
        image_size: tuple[int, int] | None = None,
    ) -> tuple[bool, str | None]:
        """Validate points in their current TL, TR, BR, BL interaction order."""
        array = np.asarray(tuple(points), dtype=np.float32)
        if array.shape != (4, 2) or not np.isfinite(array).all():
            return False, "Four valid corner positions are required."
        distances = [
            float(np.linalg.norm(array[i] - array[j]))
            for i in range(4)
            for j in range(i + 1, 4)
        ]
        if min(distances) < 1.0:
            return False, "Corner handles cannot overlap."
        if self._segments_intersect(array[0], array[1], array[2], array[3]) or self._segments_intersect(
            array[1], array[2], array[3], array[0]
        ):
            return False, "Corner lines cannot cross."
        contour = array.reshape(-1, 1, 2)
        area = abs(float(cv2.contourArea(contour)))
        if area < self.config.minimum_area_px:
            return False, "The selected card area is too small."
        if not cv2.isContourConvex(array.astype(np.int32)):
            return False, "Corners must form a convex four-sided shape."
        edges = [float(np.linalg.norm(array[(i + 1) % 4] - array[i])) for i in range(4)]
        if min(edges) < self.config.minimum_edge_px:
            return False, "Each card edge must have a visible length."
        if image_size is not None:
            width, height = image_size
            if np.any(array[:, 0] < 0) or np.any(array[:, 0] > width - 1) or np.any(array[:, 1] < 0) or np.any(array[:, 1] > height - 1):
                return False, "Corners must stay inside the source image."
        return True, None

    def _output_size(self, points: np.ndarray) -> tuple[int, int]:
        source_width = (
            float(np.linalg.norm(points[1] - points[0]))
            + float(np.linalg.norm(points[2] - points[3]))
        ) / 2
        quality_width = max(
            self.config.minimum_width_px,
            int(round(source_width * self.config.max_upscale_factor)),
        )
        width = min(
            self.config.maximum_width_px,
            self.config.preferred_width_px,
            quality_width,
        )
        width = max(64, width)
        height = max(40, int(round(width / self.target_ratio)))
        width = max(64, int(round(height * self.target_ratio)))
        return width, height

    @staticmethod
    def _residual_top_frame_metrics(
        rectified: np.ndarray,
        background_color: tuple[int, int, int] | None,
    ) -> dict[str, float]:
        if background_color is None or rectified.size == 0:
            return {
                "strip_rows": 0.0,
                "top_white_ratio": 0.0,
                "outer_score": 0.0,
                "inner_score": 0.0,
                "frame_penalty": 0.0,
            }
        height, width = rectified.shape[:2]
        scan_rows = max(3, min(int(round(height * 0.04)), 48))
        top = rectified[:scan_rows]
        lab = cv2.cvtColor(top, cv2.COLOR_RGB2LAB).astype(np.float32)
        background_lab = cv2.cvtColor(
            np.uint8([[background_color]]), cv2.COLOR_RGB2LAB
        )[0, 0].astype(np.float32)
        similar = np.linalg.norm(lab - background_lab, axis=2) <= 20.0
        gray = cv2.cvtColor(top, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 35, 110) > 0
        strip_rows = 0
        for row in range(scan_rows):
            if (
                float(np.mean(similar[row])) < 0.88
                or float(np.var(gray[row])) > 50.0
                or float(np.mean(edges[row])) > 0.045
            ):
                break
            strip_rows += 1
        top_white_ratio = float(np.mean(similar[: max(1, min(5, scan_rows))]))
        inner_start = min(scan_rows - 1, strip_rows + 1)
        inner = slice(inner_start, scan_rows)
        inner_score = float(
            np.clip(
                0.58 * (1.0 - np.mean(similar[inner]))
                + 0.42 * min(1.0, np.mean(edges[inner]) / 0.04),
                0.0,
                1.0,
            )
        )
        outer_score = float(
            np.clip(
                0.65 * top_white_ratio
                + 0.35 * (1.0 - min(1.0, np.mean(edges[: max(1, strip_rows)]) / 0.04)),
                0.0,
                1.0,
            )
        )
        return {
            "strip_rows": float(strip_rows),
            "top_white_ratio": top_white_ratio,
            "outer_score": outer_score,
            "inner_score": inner_score,
            "frame_penalty": outer_score * (1.0 - 0.35 * inner_score),
        }

    def _quality(
        self,
        points: np.ndarray,
        image_size: tuple[int, int],
        detector_confidence: float,
        inferred_count: int,
        outside_fraction: float,
        method: str,
        refinement_confidence: float,
        refinement_accepted: bool,
    ) -> tuple[float, str, str, dict[str, float]]:
        edges = np.asarray(
            [np.linalg.norm(points[(i + 1) % 4] - points[i]) for i in range(4)],
            dtype=np.float64,
        )
        opposite_balance = min(edges[0], edges[2]) / max(edges[0], edges[2])
        side_balance = min(edges[1], edges[3]) / max(edges[1], edges[3])
        distortion_score = float(np.clip(min(opposite_balance, side_balance) / 0.32, 0.0, 1.0))
        angles = [self._corner_angle(points[(i - 1) % 4], points[i], points[(i + 1) % 4]) for i in range(4)]
        minimum_angle = min(angles)
        maximum_angle = max(angles)
        angle_score = float(
            np.clip(min(minimum_angle / 38.0, (180.0 - maximum_angle) / 38.0), 0.0, 1.0)
        )
        width, height = image_size
        area_fraction = abs(float(cv2.contourArea(points))) / max(1.0, width * height)
        area_score = float(np.clip(area_fraction / 0.015, 0.0, 1.0))
        geometry_score = 0.50 * angle_score + 0.30 * distortion_score + 0.20 * area_score
        boundary_excess = max(
            0.0, outside_fraction - self.config.boundary_tolerance_fraction
        )
        boundary_score = float(np.clip(1.0 - boundary_excess / 0.08, 0.0, 1.0))
        if method == "manual":
            confidence = 0.78 * geometry_score + 0.22 * boundary_score
            confidence = max(confidence, 0.76)
        else:
            confidence = (
                0.38 * float(np.clip(detector_confidence, 0.0, 1.0))
                + 0.27 * geometry_score
                + 0.18 * boundary_score
                + 0.17 * float(np.clip(refinement_confidence, 0.0, 1.0))
                - min(0.24, inferred_count * 0.12)
            )
        confidence = float(np.clip(confidence, 0.0, 1.0))
        automatic_source_is_high = method == "manual" or detector_confidence >= self.config.high_confidence
        if (
            confidence >= self.config.high_confidence
            and automatic_source_is_high
            and inferred_count == 0
            and outside_fraction <= self.config.boundary_tolerance_fraction
            and refinement_accepted
            and refinement_confidence >= 0.56
        ):
            level, status = "high", "corrected"
        elif confidence >= self.config.medium_confidence:
            level, status = "medium", "review"
        else:
            level, status = "low", "review"
        return confidence, level, status, {
            "angle_score": round(angle_score, 4),
            "distortion_score": round(distortion_score, 4),
            "area_score": round(area_score, 4),
            "boundary_score": round(boundary_score, 4),
            "minimum_angle_degrees": round(minimum_angle, 2),
            "maximum_angle_degrees": round(maximum_angle, 2),
        }

    def _clamp_to_image(
        self, points: np.ndarray, width: int, height: int
    ) -> tuple[np.ndarray, float]:
        clamped = points.copy()
        overflow_x = np.maximum(0.0, -clamped[:, 0]) + np.maximum(0.0, clamped[:, 0] - (width - 1))
        overflow_y = np.maximum(0.0, -clamped[:, 1]) + np.maximum(0.0, clamped[:, 1] - (height - 1))
        outside_fraction = float(
            max(
                float(overflow_x.max(initial=0.0)) / max(1, width),
                float(overflow_y.max(initial=0.0)) / max(1, height),
            )
        )
        clamped[:, 0] = np.clip(clamped[:, 0], 0, width - 1)
        clamped[:, 1] = np.clip(clamped[:, 1], 0, height - 1)
        return clamped, outside_fraction

    @staticmethod
    def _corner_angle(previous: np.ndarray, center: np.ndarray, following: np.ndarray) -> float:
        first = previous - center
        second = following - center
        denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-9)
        cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
        return degrees(acos(cosine))

    @staticmethod
    def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
        def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
            first = q - p
            second = r - p
            return float(first[0] * second[1] - first[1] * second[0])

        return orientation(a, b, c) * orientation(a, b, d) < 0 and orientation(c, d, a) * orientation(c, d, b) < 0

    @staticmethod
    def _point_tuple(points: np.ndarray) -> tuple[Point, ...]:
        return tuple((float(point[0]), float(point[1])) for point in points)
