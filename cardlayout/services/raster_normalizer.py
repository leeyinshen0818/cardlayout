from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass(slots=True)
class NormalizedRaster:
    image: Image.Image
    original_size: tuple[int, int]
    trim_applied: bool = False
    trim_box: tuple[int, int, int, int] | None = None
    outer_background_mask: Image.Image | None = None
    trim_overlay: Image.Image | None = None
    background_color: tuple[int, int, int] | None = None
    background_tolerance: float | None = None
    background_confidence: float = 0.0
    safety_margin_fraction_used: float | None = None
    initial_trim_box: tuple[int, int, int, int] | None = None
    second_pass_trim_box: tuple[int, int, int, int] | None = None
    first_trimmed_image: Image.Image | None = None
    residual_trim_overlay: Image.Image | None = None
    residual_trim_offsets: tuple[int, int, int, int] = (0, 0, 0, 0)
    residual_metrics: dict[str, float] | None = None


def normalize_raster_for_detection(
    image: Image.Image,
    *,
    trim_pdf_whitespace: bool = False,
    trim_safety_margin_fraction: float = 0.025,
) -> NormalizedRaster:
    """Return the canonical RGB uint8 raster used by every processing stage."""
    transposed = ImageOps.exif_transpose(image)
    original_size = transposed.size
    if "A" in transposed.getbands() or transposed.mode == "P":
        rgba = transposed.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        canonical = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        canonical = transposed.convert("RGB")

    background = (
        _border_connected_background(canonical, trim_safety_margin_fraction)
        if trim_pdf_whitespace
        else None
    )
    initial_trim_box = background.trim_box if background is not None else None
    trim_box = initial_trim_box
    untrimmed = canonical.copy()
    first_trimmed = canonical.crop(trim_box) if trim_box is not None else canonical.copy()
    residual_overlay = first_trimmed.copy()
    residual_offsets = (0, 0, 0, 0)
    residual_metrics: dict[str, float] | None = None
    second_pass_trim_box = trim_box
    if background is not None and trim_box is not None:
        residual = _iterative_residual_border_cleanup(
            first_trimmed,
            background.color,
            background.tolerance,
            maximum_iterations=2,
        )
        residual_offsets = residual.offsets
        residual_metrics = residual.metrics
        residual_overlay = residual.overlay
        if any(residual.offsets):
            first_left, first_top, _, _ = trim_box
            relative_left, relative_top, relative_right, relative_bottom = residual.box
            trim_box = (
                first_left + relative_left,
                first_top + relative_top,
                first_left + relative_right,
                first_top + relative_bottom,
            )
            second_pass_trim_box = trim_box
    if trim_box is not None:
        canonical = canonical.crop(trim_box)
    return NormalizedRaster(
        image=canonical.copy(),
        original_size=original_size,
        trim_applied=trim_box is not None,
        trim_box=trim_box,
        outer_background_mask=(background.mask if background is not None else None),
        trim_overlay=(
            _trim_overlay(untrimmed, trim_box) if background is not None else None
        ),
        background_color=(background.color if background is not None else None),
        background_tolerance=(
            background.tolerance if background is not None else None
        ),
        background_confidence=(
            background.confidence if background is not None else 0.0
        ),
        safety_margin_fraction_used=(
            background.safety_margin_fraction if background is not None else None
        ),
        initial_trim_box=initial_trim_box,
        second_pass_trim_box=second_pass_trim_box,
        first_trimmed_image=first_trimmed,
        residual_trim_overlay=residual_overlay,
        residual_trim_offsets=residual_offsets,
        residual_metrics=residual_metrics,
    )


@dataclass(slots=True)
class _BackgroundDetection:
    trim_box: tuple[int, int, int, int] | None
    mask: Image.Image
    color: tuple[int, int, int]
    tolerance: float
    confidence: float
    safety_margin_fraction: float


def _border_connected_background(
    image: Image.Image,
    safety_margin_fraction: float,
) -> _BackgroundDetection | None:
    """Find low-texture paper color connected to the outside of a PDF raster."""
    original_width, original_height = image.size
    analysis = image.copy()
    analysis.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
    rgb = np.asarray(analysis, dtype=np.uint8)
    height, width = rgb.shape[:2]
    if min(width, height) < 160:
        return None
    strip = max(3, int(round(min(width, height) * 0.015)))
    border = np.concatenate(
        (
            rgb[:strip].reshape(-1, 3),
            rgb[-strip:].reshape(-1, 3),
            rgb[strip:-strip, :strip].reshape(-1, 3),
            rgb[strip:-strip, -strip:].reshape(-1, 3),
        ),
        axis=0,
    )
    border_hsv = cv2.cvtColor(
        border.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV
    ).reshape(-1, 3)
    border_gray = cv2.cvtColor(
        border.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY
    ).reshape(-1)
    border_color = np.median(border.astype(np.float32), axis=0)
    if (
        float(np.median(border_gray)) < 185.0
        or float(np.median(border_hsv[:, 1])) > 48.0
        or float(np.std(border_gray)) > 13.0
    ):
        return None

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    background_lab = np.median(
        cv2.cvtColor(
            border.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
        ).reshape(-1, 3).astype(np.float32),
        axis=0,
    )
    border_distance = np.linalg.norm(
        cv2.cvtColor(border.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB)
        .reshape(-1, 3)
        .astype(np.float32)
        - background_lab,
        axis=1,
    )
    tolerance = float(np.clip(np.percentile(border_distance, 97) + 7.0, 9.0, 25.0))
    color_distance = np.linalg.norm(lab - background_lab, axis=2)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    border_gradient = np.concatenate(
        (
            gradient[:strip].ravel(),
            gradient[-strip:].ravel(),
            gradient[strip:-strip, :strip].ravel(),
            gradient[strip:-strip, -strip:].ravel(),
        )
    )
    gradient_limit = float(np.clip(np.percentile(border_gradient, 97) + 12.0, 18.0, 55.0))
    candidate = ((color_distance <= tolerance) & (gradient <= gradient_limit)).astype(
        np.uint8
    )
    label_count, labels = cv2.connectedComponents(candidate, connectivity=8)
    if label_count <= 1:
        return None
    boundary_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    boundary_labels = boundary_labels[boundary_labels != 0]
    connected_background = np.isin(labels, boundary_labels)
    boundary_coverage = float(
        np.mean(
            np.concatenate(
                (
                    connected_background[0],
                    connected_background[-1],
                    connected_background[:, 0],
                    connected_background[:, -1],
                )
            )
        )
    )
    background_fraction = float(np.mean(connected_background))
    raw_mask = (connected_background.astype(np.uint8) * 255)
    mask_image = Image.fromarray(raw_mask).resize(
        (original_width, original_height), Image.Resampling.NEAREST
    )
    uniformity = float(np.clip(1.0 - np.std(border_gray) / 13.0, 0.0, 1.0))
    confidence = float(
        np.clip(
            0.48 * boundary_coverage
            + 0.27 * uniformity
            + 0.25 * min(1.0, background_fraction / 0.10),
            0.0,
            1.0,
        )
    )
    detected = _BackgroundDetection(
        trim_box=None,
        mask=mask_image,
        color=tuple(int(round(value)) for value in border_color),
        tolerance=round(tolerance, 3),
        confidence=confidence,
        safety_margin_fraction=safety_margin_fraction,
    )
    if (
        boundary_coverage < 0.90
        or confidence < 0.82
        or not 0.025 <= background_fraction <= 0.995
    ):
        return detected

    foreground = (~connected_background).astype(np.uint8)
    component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground, connectivity=8
    )
    minimum_component_area = max(20, int(round(width * height * 0.00005)))
    kept = np.zeros_like(foreground)
    for label in range(1, component_count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_component_area:
            kept[component_labels == label] = 255
    points = cv2.findNonZero(kept)
    if points is None:
        return detected
    x, y, content_width, content_height = cv2.boundingRect(points)
    content_area_ratio = (content_width * content_height) / float(width * height)
    margins = (x, y, width - (x + content_width), height - (y + content_height))
    meaningful_margin = min(width, height) * 0.012
    if content_area_ratio > 0.94 or sum(value >= meaningful_margin for value in margins) < 3:
        return detected

    content_ratio = max(content_width, content_height) / max(
        1.0, min(content_width, content_height)
    )
    effective_safety_fraction = (
        min(safety_margin_fraction, 0.005)
        if 1.45 <= content_ratio <= 1.82
        else safety_margin_fraction
    )
    detected.safety_margin_fraction = effective_safety_fraction
    safety = max(8, int(round(min(width, height) * effective_safety_fraction)))
    left = max(0, x - safety)
    top = max(0, y - safety)
    right = min(width, x + content_width + safety)
    bottom = min(height, y + content_height + safety)
    if (right - left) * (bottom - top) >= width * height * 0.97:
        return detected
    scale_x = original_width / width
    scale_y = original_height / height
    detected.trim_box = (
        max(0, int(np.floor(left * scale_x))),
        max(0, int(np.floor(top * scale_y))),
        min(original_width, int(np.ceil(right * scale_x))),
        min(original_height, int(np.ceil(bottom * scale_y))),
    )
    return detected


def _trim_overlay(
    image: Image.Image, trim_box: tuple[int, int, int, int] | None
) -> Image.Image:
    overlay = np.asarray(image).copy()
    if trim_box is not None:
        left, top, right, bottom = trim_box
        thickness = max(2, int(round(min(image.size) * 0.002)))
        cv2.rectangle(
            overlay,
            (left, top),
            (max(left, right - 1), max(top, bottom - 1)),
            (20, 220, 80),
            thickness,
            cv2.LINE_AA,
        )
    return Image.fromarray(overlay)


@dataclass(slots=True)
class _ResidualCleanup:
    box: tuple[int, int, int, int]
    offsets: tuple[int, int, int, int]
    metrics: dict[str, float]
    overlay: Image.Image


def _iterative_residual_border_cleanup(
    image: Image.Image,
    background_color: tuple[int, int, int],
    background_tolerance: float,
    *,
    maximum_iterations: int,
) -> _ResidualCleanup:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    current = (0, 0, width, height)
    total_offsets = [0, 0, 0, 0]  # top, right, bottom, left
    first_metrics: dict[str, float] | None = None
    for _ in range(maximum_iterations):
        left, top, right, bottom = current
        view = rgb[top:bottom, left:right]
        offsets, metrics = _residual_band_offsets(
            view, background_color, background_tolerance
        )
        if first_metrics is None:
            first_metrics = metrics
        if not any(offsets):
            break
        trim_top, trim_right, trim_bottom, trim_left = offsets
        current = (
            left + trim_left,
            top + trim_top,
            right - trim_right,
            bottom - trim_bottom,
        )
        total_offsets[0] += trim_top
        total_offsets[1] += trim_right
        total_offsets[2] += trim_bottom
        total_offsets[3] += trim_left
        if current[2] - current[0] < 80 or current[3] - current[1] < 50:
            current = (left, top, right, bottom)
            break

    overlay = rgb.copy()
    if any(total_offsets):
        cv2.rectangle(
            overlay,
            (current[0], current[1]),
            (current[2] - 1, current[3] - 1),
            (245, 155, 25),
            max(2, int(round(min(width, height) * 0.002))),
            cv2.LINE_AA,
        )
    metrics = first_metrics or {
        "top_border_mean": 0.0,
        "top_border_variance": 0.0,
        "top_border_edge_density": 0.0,
        "top_candidate_outer_score": 0.0,
        "top_candidate_inner_score": 0.0,
        "pdf_frame_edge_penalty": 0.0,
    }
    metrics["selected_top_edge_offset"] = float(total_offsets[0])
    return _ResidualCleanup(
        box=current,
        offsets=tuple(total_offsets),  # type: ignore[arg-type]
        metrics=metrics,
        overlay=Image.fromarray(overlay),
    )


def _residual_band_offsets(
    rgb: np.ndarray,
    background_color: tuple[int, int, int],
    background_tolerance: float,
) -> tuple[tuple[int, int, int, int], dict[str, float]]:
    height, width = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    background_lab = cv2.cvtColor(
        np.uint8([[background_color]]), cv2.COLOR_RGB2LAB
    )[0, 0].astype(np.float32)
    distance = np.linalg.norm(lab - background_lab, axis=2)
    similar = distance <= min(30.0, background_tolerance + 5.0)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 35, 110) > 0
    maximum_y = max(3, min(int(round(height * 0.035)), 60))
    maximum_x = max(3, min(int(round(width * 0.035)), 60))

    def removable_count(
        similarity: np.ndarray, values: np.ndarray, edge_values: np.ndarray
    ) -> int:
        length = similarity.shape[0]
        count = 0
        for index in range(length):
            coverage = float(np.mean(similarity[index]))
            variance = float(np.var(values[index]))
            edge_density = float(np.mean(edge_values[index]))
            if coverage < 0.82 or variance > 45.0 or edge_density > 0.045:
                break
            count += 1
        if count < 3:
            return 0
        inner_start = min(length, count + 1)
        inner_end = min(length, count + max(5, length // 5))
        if inner_end <= inner_start:
            return 0
        inner_similarity = float(np.mean(similarity[inner_start:inner_end]))
        inner_edges = float(np.mean(edge_values[inner_start:inner_end]))
        if inner_similarity > 0.62 and inner_edges < 0.012:
            return 0
        return max(0, count - 2)

    top_count = removable_count(
        similar[:maximum_y], gray[:maximum_y], edges[:maximum_y]
    )
    bottom_count = removable_count(
        similar[-maximum_y:][::-1], gray[-maximum_y:][::-1], edges[-maximum_y:][::-1]
    )
    left_count = removable_count(
        similar[:, :maximum_x].T, gray[:, :maximum_x].T, edges[:, :maximum_x].T
    )
    right_count = removable_count(
        similar[:, -maximum_x:][:, ::-1].T,
        gray[:, -maximum_x:][:, ::-1].T,
        edges[:, -maximum_x:][:, ::-1].T,
    )
    top_band = gray[: max(1, min(maximum_y, max(3, top_count + 2)))]
    top_edges = edges[: top_band.shape[0]]
    top_similarity = similar[: top_band.shape[0]]
    outer_score = float(
        np.clip(
            0.55 * np.mean(top_similarity)
            + 0.25 * (1.0 - min(1.0, np.var(top_band) / 45.0))
            + 0.20 * (1.0 - min(1.0, np.mean(top_edges) / 0.045)),
            0.0,
            1.0,
        )
    )
    inner_start = min(height - 1, top_count + 2)
    inner_end = min(height, inner_start + max(5, maximum_y // 2))
    inner_score = float(
        np.clip(
            0.55 * (1.0 - np.mean(similar[inner_start:inner_end]))
            + 0.45 * min(1.0, np.mean(edges[inner_start:inner_end]) / 0.04),
            0.0,
            1.0,
        )
    )
    metrics = {
        "top_border_mean": round(float(np.mean(top_band)), 3),
        "top_border_variance": round(float(np.var(top_band)), 3),
        "top_border_edge_density": round(float(np.mean(top_edges)), 5),
        "top_candidate_outer_score": round(outer_score, 4),
        "top_candidate_inner_score": round(inner_score, 4),
        "pdf_frame_edge_penalty": round(outer_score * (1.0 - inner_score * 0.35), 4),
    }
    return (top_count, right_count, bottom_count, left_count), metrics
