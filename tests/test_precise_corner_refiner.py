from __future__ import annotations

import cv2
import numpy as np
import pytest

from PIL import Image

from cardlayout.models.card_size import MALAYSIA_IC
from cardlayout.models.perspective import EdgeFitResult, PerspectiveConfig
from cardlayout.services.perspective_corrector import PerspectiveCorrector

TARGET_RATIO = MALAYSIA_IC.width_mm / MALAYSIA_IC.height_mm
from cardlayout.services.precise_corner_refiner import PreciseCornerRefiner


def _rounded_mask(width: int, height: int, radius: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (width - radius - 1, height - 1), 255, -1)
    cv2.rectangle(mask, (0, radius), (width - 1, height - radius - 1), 255, -1)
    for center in (
        (radius, radius),
        (width - radius - 1, radius),
        (width - radius - 1, height - radius - 1),
        (radius, height - radius - 1),
    ):
        cv2.circle(mask, center, radius, 255, -1, cv2.LINE_AA)
    return mask


def _card_scene(
    *,
    partial_left_occlusion: bool = False,
    strong_perspective: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    canvas = np.full((900, 1200, 3), (35, 41, 50), dtype=np.uint8)
    cv2.rectangle(canvas, (725, 65), (1130, 315), (218, 225, 232), -1)
    for y in range(710, 880, 24):
        cv2.line(canvas, (520, y), (1170, y - 28), (78, 86, 98), 5)

    card_width, card_height = 856, 540
    card = np.full((card_height, card_width, 3), (188, 220, 235), dtype=np.uint8)
    for x in range(card_width):
        card[:, x, 1] = np.clip(190 + x * 38 / card_width, 0, 255)
    cv2.rectangle(card, (95, 135), (245, 285), (210, 178, 65), 8)
    cv2.rectangle(card, (605, 115), (790, 390), (55, 80, 120), 7)
    cv2.line(card, (70, 410), (760, 410), (50, 95, 145), 5)
    mask = _rounded_mask(card_width, card_height, 34)
    source = np.asarray(
        ((0, 0), (card_width - 1, 0), (card_width - 1, card_height - 1), (0, card_height - 1)),
        dtype=np.float32,
    )
    if strong_perspective:
        true = np.asarray(
            ((310, 120), (845, 205), (1025, 690), (130, 755)), dtype=np.float32
        )
    else:
        true = np.asarray(
            ((235, 190), (900, 145), (970, 660), (170, 710)), dtype=np.float32
        )
    transform = cv2.getPerspectiveTransform(source, true)
    warped_card = cv2.warpPerspective(card, transform, (1200, 900), flags=cv2.INTER_CUBIC)
    warped_mask = cv2.warpPerspective(mask, transform, (1200, 900), flags=cv2.INTER_LINEAR)
    alpha = warped_mask.astype(np.float32)[:, :, None] / 255.0
    canvas = np.clip(warped_card * alpha + canvas * (1.0 - alpha), 0, 255).astype(np.uint8)
    if partial_left_occlusion:
        start = tuple(np.round(true[0] * 0.63 + true[3] * 0.37).astype(int))
        end = tuple(np.round(true[0] * 0.31 + true[3] * 0.69).astype(int))
        cv2.line(canvas, start, end, (186, 133, 105), 62, cv2.LINE_AA)
    rough = true + np.asarray(((-9, -7), (10, -8), (11, 9), (-10, 10)), np.float32)
    return canvas, true, rough


def test_line_intersection_is_precise_and_parallel_lines_are_rejected() -> None:
    refiner = PreciseCornerRefiner()
    horizontal = np.asarray((0.0, 1.0, -42.0))
    vertical = np.asarray((1.0, 0.0, -73.0))
    point = refiner.intersect_lines(horizontal, vertical)
    assert point is not None
    assert tuple(point) == pytest.approx((73.0, 42.0))
    assert refiner.intersect_lines(horizontal, np.asarray((0.0, 1.0, -80.0))) is None


def test_search_band_is_local_and_excludes_internal_graphics() -> None:
    refiner = PreciseCornerRefiner()
    mask, polygon = refiner.build_search_band(
        (500, 800), np.asarray((100.0, 120.0)), np.asarray((700.0, 120.0)), 20.0
    )
    assert polygon.shape == (4, 2)
    assert mask[120, 400] == 255
    assert mask[155, 400] == 0
    assert mask[120, 105] == 0  # Rounded-corner endpoint area is trimmed.


def test_ransac_rejects_many_outliers_and_recovers_distributed_line() -> None:
    rng = np.random.default_rng(9)
    x = np.linspace(40, 760, 220)
    line_points = np.column_stack((x, 0.22 * x + 85 + rng.normal(0, 0.55, len(x))))
    outliers = rng.uniform((20, 30), (790, 450), size=(320, 2))
    points = np.vstack((line_points, outliers))
    weights = np.concatenate((np.ones(len(line_points)), np.full(len(outliers), 0.42)))
    refiner = PreciseCornerRefiner(PerspectiveConfig(ransac_iterations=100))
    result, inliers = refiner.fit_line_ransac(
        "top",
        points,
        weights,
        np.asarray((40.0, 94.0)),
        np.asarray((760.0, 252.0)),
        24.0,
        2.0,
    )
    assert result.success and result.line is not None
    assert result.support_ratio > 0.85
    assert result.inlier_count > 190
    assert inliers[: len(line_points)].mean() > 0.9
    recovered = np.asarray(result.line)
    expected = refiner.line_from_points(np.asarray((40.0, 93.8)), np.asarray((760.0, 252.2)))
    assert refiner.line_angle_delta(recovered, expected) < 0.3


@pytest.mark.parametrize("strong_perspective", [False, True])
def test_rounded_card_edges_refine_to_mathematical_intersections(
    strong_perspective: bool,
) -> None:
    image, true, rough = _card_scene(strong_perspective=strong_perspective)
    result = PreciseCornerRefiner().refine(image, rough)
    refined = np.asarray(result.refined_corners)
    rough_error = np.linalg.norm(rough - true, axis=1)
    refined_error = np.linalg.norm(refined - true, axis=1)

    assert result.success, result.fallback_reason
    assert all(not edge.inferred for edge in result.edge_results)
    assert refined_error.mean() < rough_error.mean() * 0.55
    assert refined_error.max() < 6.0
    assert min(result.corner_confidences) >= 0.43


def test_partial_finger_occlusion_uses_separated_edge_support() -> None:
    image, true, rough = _card_scene(partial_left_occlusion=True)
    result = PreciseCornerRefiner().refine(image, rough)
    refined = np.asarray(result.refined_corners)
    left = result.edge_results[3]

    assert result.success, result.fallback_reason
    assert left.support_ratio > 0.62
    assert left.support_length_px > 250
    assert np.linalg.norm(refined - true, axis=1).mean() < 7.0


def test_one_missing_boundary_retains_rough_edge_and_reduces_confidence() -> None:
    image = np.full((650, 900, 3), (35, 42, 50), dtype=np.uint8)
    image[0:510, 150:750] = (190, 220, 235)
    rough = np.asarray(((150, 170), (750, 170), (750, 510), (150, 510)), np.float32)
    result = PreciseCornerRefiner().refine(image, rough)

    top = result.edge_results[0]
    assert result.success
    assert top.inferred
    assert sum(edge.inferred for edge in result.edge_results) == 1
    assert result.fallback_reason == "1 edge(s) retained rough geometry."
    assert result.confidence < 0.75


def test_color_only_boundary_is_detected_without_brightness_polarity_assumption() -> None:
    image = np.full((650, 900, 3), (0, 130, 0), dtype=np.uint8)
    cv2.rectangle(image, (150, 150), (750, 528), (255, 0, 0), -1)
    rough = np.asarray(((144, 144), (756, 144), (756, 534), (144, 534)), np.float32)
    result = PreciseCornerRefiner().refine(image, rough)
    refined = np.asarray(result.refined_corners)
    true = np.asarray(((150, 150), (750, 150), (750, 528), (150, 528)), np.float32)

    assert result.success, result.fallback_reason
    assert np.linalg.norm(refined - true, axis=1).max() < 4.0


def test_internal_chip_and_portrait_rectangles_do_not_hijack_outer_edges() -> None:
    image, true, rough = _card_scene()
    result = PreciseCornerRefiner().refine(image, rough)
    refined = np.asarray(result.refined_corners)
    assert result.success
    assert np.linalg.norm(refined - true, axis=1).max() < 6.0
    assert all(edge.rough_offset_px < 18 for edge in result.edge_results)


def test_flat_roi_falls_back_to_rough_geometry_with_low_confidence() -> None:
    image = np.full((700, 1000, 3), 128, dtype=np.uint8)
    rough = np.asarray(((120, 100), (850, 110), (830, 580), (105, 570)), np.float32)
    result = PreciseCornerRefiner().refine(image, rough)
    assert not result.success
    assert result.refined_corners == result.rough_corners
    assert result.confidence <= 0.34
    assert result.fallback_reason


def test_extreme_corner_displacement_is_rejected() -> None:
    refiner = PreciseCornerRefiner()
    rough = np.asarray(((100, 100), (800, 100), (800, 540), (100, 540)), np.float32)
    moved = rough.copy()
    moved[0] += (-70, -60)
    valid, reason = refiner._validate_refinement(rough, moved)
    assert not valid
    assert "too far" in (reason or "").lower()


def test_debug_mode_exposes_rough_bands_evidence_fits_and_metrics() -> None:
    image, _, rough = _card_scene()
    result = PreciseCornerRefiner(debug=True).refine(image, rough)
    assert result.success
    assert set(result.debug_info["stage_images"]) == {
        "rough_and_search_bands",
        "raw_edge_evidence",
        "fitted_edges_and_intersections",
    }
    assert set(result.debug_info["edge_scores"]) == {"top", "right", "bottom", "left"}
    assert len(result.debug_info["rough_to_refined_corner_distances"]) == 4


def _weak_blue_back(
    *,
    internal_rectangle: bool = False,
    occlusion: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.full((820, 1200, 3), (43, 75, 108), dtype=np.uint8)
    true = np.asarray(((200, 165), (1000, 165), (1000, 669), (200, 669)), np.float32)
    cv2.rectangle(image, (200, 165), (1000, 669), (50, 86, 124), -1)
    cv2.rectangle(image, (200, 165), (1000, 669), (57, 95, 134), 2, cv2.LINE_AA)
    if internal_rectangle:
        cv2.rectangle(image, (230, 195), (970, 639), (220, 230, 240), 7, cv2.LINE_AA)
    else:
        cv2.circle(image, (270, 235), 18, (61, 101, 143), -1, cv2.LINE_AA)
    skin = (184, 132, 106)
    if occlusion == "edge":
        cv2.line(image, (450, 160), (690, 168), skin, 66, cv2.LINE_AA)
    elif occlusion == "corner":
        cv2.circle(image, (205, 170), 78, skin, -1, cv2.LINE_AA)
    elif occlusion == "adjacent":
        cv2.line(image, (195, 185), (205, 390), skin, 62, cv2.LINE_AA)
        cv2.line(image, (205, 170), (420, 165), skin, 62, cv2.LINE_AA)
    rough = true + np.asarray(((-5, -4), (5, -4), (5, 5), (-5, 5)), np.float32)
    return image, true, rough


def _quad_area(points: np.ndarray) -> float:
    return abs(float(cv2.contourArea(np.asarray(points, dtype=np.float32))))


def test_low_texture_uniform_blue_back_never_collapses() -> None:
    image, _, rough = _weak_blue_back()
    result = PreciseCornerRefiner(target_ratio=TARGET_RATIO).refine(image, rough)
    refined = np.asarray(result.refined_corners)

    assert _quad_area(refined) / _quad_area(rough) >= 0.90
    assert np.linalg.norm(refined.mean(axis=0) - rough.mean(axis=0)) < 12
    assert tuple(edge.name for edge in result.edge_results) == (
        "top", "right", "bottom", "left"
    )
    assert len(result.edge_confidences) == 4


@pytest.mark.parametrize("occlusion", ["edge", "corner", "adjacent"])
def test_partial_hand_occlusion_reduces_confidence_without_inward_collapse(
    occlusion: str,
) -> None:
    image, _, rough = _weak_blue_back(occlusion=occlusion)
    result = PreciseCornerRefiner(target_ratio=TARGET_RATIO).refine(image, rough)
    refined = np.asarray(result.refined_corners)
    clear_image, _, clear_rough = _weak_blue_back()
    clear = PreciseCornerRefiner(target_ratio=TARGET_RATIO).refine(
        clear_image, clear_rough
    )

    assert _quad_area(refined) / _quad_area(rough) >= 0.88
    assert max(np.linalg.norm(refined - rough, axis=1)) < 55
    assert result.confidence < clear.confidence


def test_strong_internal_printed_rectangle_cannot_replace_weak_outer_boundary() -> None:
    image, _, rough = _weak_blue_back(internal_rectangle=True)
    result = PreciseCornerRefiner(target_ratio=TARGET_RATIO).refine(image, rough)
    refined = np.asarray(result.refined_corners)

    assert _quad_area(refined) / _quad_area(rough) >= 0.90
    assert np.linalg.norm(refined.mean(axis=0) - rough.mean(axis=0)) < 12
    if not result.success:
        assert result.refined_corners == result.rough_corners


def test_logo_text_and_chip_edges_do_not_dominate_outer_boundary() -> None:
    image, true, rough = _weak_blue_back()
    cv2.rectangle(image, (250, 225), (430, 390), (225, 230, 235), 9)
    cv2.rectangle(image, (760, 220), (925, 520), (25, 45, 70), 8)
    for y in range(470, 590, 22):
        cv2.line(image, (285, y), (700, y), (220, 225, 230), 6)
    result = PreciseCornerRefiner(target_ratio=TARGET_RATIO).refine(image, rough)
    refined = np.asarray(result.refined_corners)

    assert _quad_area(refined) / _quad_area(rough) >= 0.90
    assert np.linalg.norm(refined.mean(axis=0) - true.mean(axis=0)) < 12


def test_incorrect_inward_refinement_is_rejected_without_strong_outer_evidence() -> None:
    refiner = PreciseCornerRefiner(target_ratio=TARGET_RATIO)
    rough = np.asarray(((100, 100), (900, 100), (900, 604), (100, 604)), np.float32)
    inward = np.asarray(((132, 132), (868, 132), (868, 572), (132, 572)), np.float32)
    weak_internal_edges = [
        EdgeFitResult(
            name=name,
            success=True,
            score=0.57,
            support_ratio=0.62,
            signed_rough_offset_px=-32.0,
        )
        for name in ("top", "right", "bottom", "left")
    ]

    valid, reason = refiner._validate_refinement(rough, inward, weak_internal_edges)

    assert not valid
    assert "collapse" in (reason or "").lower()


def test_failed_automatic_refinement_preserves_rough_quad_and_requests_review() -> None:
    image = np.full((820, 1200, 3), (45, 78, 112), dtype=np.uint8)
    rough = np.asarray(((200, 165), (1000, 165), (1000, 669), (200, 669)), np.float32)
    cv2.rectangle(image, (230, 195), (970, 639), (230, 235, 240), 8)

    result = PerspectiveCorrector(MALAYSIA_IC).correct(
        Image.fromarray(image), rough, detector_confidence=0.82
    )

    assert result.success
    assert result.status == "review"
    assert np.allclose(np.asarray(result.refined_points), rough)
    assert result.refinement_fallback_reason
