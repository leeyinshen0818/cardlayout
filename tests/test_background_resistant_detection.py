from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from cardlayout.models.card_size import MALAYSIA_IC
from cardlayout.services.card_detector import CardDetector
from cardlayout.services.perspective_corrector import PerspectiveCorrector
from cardlayout.services.precise_corner_refiner import PreciseCornerRefiner


def _bounds(points: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(points[:, 0].min()),
        int(points[:, 1].min()),
        int(points[:, 0].max()),
        int(points[:, 1].max()),
    )


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / float(first_area + second_area - intersection)


def _nested_scene(
    *,
    surface: tuple[int, int, int] = (235, 235, 230),
    desk: tuple[int, int, int] = (55, 60, 65),
    extra: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.full((900, 1200, 3), desk, dtype=np.uint8)
    outer = cv2.boxPoints(((600, 450), (980, 650), 2)).astype(np.int32)
    cv2.fillConvexPoly(pixels, outer, surface)
    cv2.polylines(pixels, [outer], True, (95, 95, 92), 7)
    if extra == "notebook":
        for y in range(220, 720, 36):
            cv2.line(pixels, (170, y), (1030, y + 30), (190, 195, 190), 2)
    elif extra == "phone":
        phone = cv2.boxPoints(((970, 510), (150, 300), -8)).astype(np.int32)
        cv2.fillConvexPoly(pixels, phone, (25, 28, 35))
        cv2.polylines(pixels, [phone], True, (5, 5, 8), 6)

    card = cv2.boxPoints(((570, 470), (360, 227), -8)).astype(np.int32)
    cv2.fillConvexPoly(pixels, card, (170, 215, 235))
    cv2.polylines(pixels, [card], True, (20, 45, 70), 6)
    cv2.circle(pixels, (650, 460), 45, (80, 120, 175), -1)
    cv2.rectangle(pixels, (430, 420), (500, 480), (215, 180, 75), 5)
    for y in (520, 545):
        cv2.line(pixels, (430, y), (680, y - 35), (45, 100, 145), 4)
    return pixels, card


@pytest.mark.parametrize(
    ("surface", "desk", "extra"),
    [
        ((245, 245, 242), (55, 60, 65), "none"),
        ((210, 220, 215), (55, 60, 65), "none"),
        ((40, 45, 52), (175, 180, 185), "none"),
        ((235, 235, 230), (55, 60, 65), "phone"),
        ((235, 235, 230), (55, 60, 65), "notebook"),
    ],
    ids=("white-paper", "light-mat", "dark-mat", "phone", "notebook"),
)
def test_nested_card_beats_competing_rectangles(surface, desk, extra) -> None:
    pixels, card = _nested_scene(surface=surface, desk=desk, extra=extra)
    result = CardDetector(MALAYSIA_IC, debug=True).detect(Image.fromarray(pixels))

    assert result.success
    assert result.bounding_box is not None
    assert _iou(result.bounding_box, _bounds(card)) > 0.90
    selected = result.debug_info["selected_candidate"]
    assert selected["contained_by_candidates"]
    assert selected["interior_complexity_score"] > 0.25
    assert any(
        {"nested_parent_penalty", "oversized_background"}
        & set(candidate["rejection_reasons"])
        for candidate in result.debug_info["candidates"]
    )


def test_blank_paper_is_not_a_high_confidence_card() -> None:
    pixels = np.full((900, 1200, 3), (55, 60, 65), dtype=np.uint8)
    paper = cv2.boxPoints(((600, 450), (980, 650), 2)).astype(np.int32)
    cv2.fillConvexPoly(pixels, paper, (240, 240, 236))
    cv2.polylines(pixels, [paper], True, (90, 90, 90), 7)

    result = CardDetector(MALAYSIA_IC).detect(Image.fromarray(pixels))

    assert not result.success
    assert result.confidence_level in {"none", "low"}
    assert result.debug_info["selected_candidate"]["oversize_penalty"] > 0.10


def test_portrait_phone_scene_with_keyboard_and_cream_mat_corrects_only_card() -> None:
    pixels = np.full((1080, 608, 3), (38, 41, 47), dtype=np.uint8)
    cv2.rectangle(pixels, (0, 0), (607, 170), (30, 32, 38), -1)
    for x in (45, 155, 265, 375, 485):
        cv2.rectangle(pixels, (x, 45), (x + 70, 120), (120, 190, 45), -1)
    cv2.rectangle(pixels, (0, 165), (607, 620), (235, 230, 210), -1)
    cv2.line(pixels, (0, 620), (607, 620), (85, 80, 75), 6)

    card = cv2.boxPoints(((305, 455), (440, 278), 2)).astype(np.int32)
    cv2.fillConvexPoly(pixels, card, (165, 215, 238))
    cv2.polylines(pixels, [card], True, (15, 35, 55), 9)
    cv2.rectangle(pixels, (120, 395), (195, 460), (220, 180, 65), 5)
    cv2.circle(pixels, (410, 445), 55, (75, 110, 170), -1)
    for y in (515, 540):
        cv2.line(pixels, (110, y), (350, y + 5), (45, 95, 145), 4)

    image = Image.fromarray(pixels)
    detection = CardDetector(MALAYSIA_IC, debug=True).detect(image)
    correction = PerspectiveCorrector(MALAYSIA_IC).correct(
        image,
        detection.polygon_points,
        detector_confidence=detection.confidence,
        inferred_corner_count=detection.debug_info["inferred_edge_count"],
    )

    assert detection.success
    assert detection.bounding_box is not None
    assert _iou(detection.bounding_box, _bounds(card)) > 0.94
    assert correction.success
    assert correction.refinement_fallback_reason is None
    assert correction.refinement_confidence > 0.80
    assert correction.output_dimensions is not None
    assert correction.output_dimensions[0] / correction.output_dimensions[1] == pytest.approx(
        85.6 / 54, abs=0.01
    )


def test_refinement_roi_excludes_remote_background_edges() -> None:
    image = np.full((900, 1200, 3), (40, 45, 52), dtype=np.uint8)
    cv2.rectangle(image, (80, 70), (1120, 830), (238, 235, 220), -1)
    cv2.rectangle(image, (350, 300), (850, 615), (165, 215, 235), -1)
    cv2.rectangle(image, (350, 300), (850, 615), (20, 45, 70), 5)
    cv2.circle(image, (690, 440), 55, (70, 110, 170), -1)
    rough = np.asarray(((344, 294), (856, 294), (856, 621), (344, 621)), np.float32)

    result = PreciseCornerRefiner(target_ratio=85.6 / 54).refine(image, rough)
    refined = np.asarray(result.refined_corners)

    assert result.success, result.fallback_reason
    assert result.roi_box is not None
    assert result.roi_box[0] > 250 and result.roi_box[2] < 950
    assert np.linalg.norm(
        refined
        - np.asarray(((350, 300), (850, 300), (850, 615), (350, 615)), np.float32),
        axis=1,
    ).max() < 5.0


def test_area_expansion_is_classified_as_background_edge_hijack() -> None:
    refiner = PreciseCornerRefiner(target_ratio=85.6 / 54)
    rough = np.asarray(((100, 100), (600, 100), (600, 415), (100, 415)), np.float32)
    expanded = np.asarray(((65, 75), (635, 75), (635, 440), (65, 440)), np.float32)

    valid, reason = refiner._validate_refinement(rough, expanded)

    assert not valid
    assert reason is not None and "background_edge_hijack" in reason
    assert "area_expansion" in reason


def test_ratio_degradation_is_classified_as_background_edge_hijack() -> None:
    refiner = PreciseCornerRefiner(target_ratio=85.6 / 54)
    rough = np.asarray(((100, 100), (600, 100), (600, 415), (100, 415)), np.float32)
    worse_ratio = np.asarray(((110, 85), (590, 85), (590, 430), (110, 430)), np.float32)

    valid, reason = refiner._validate_refinement(rough, worse_ratio)

    assert not valid
    assert reason is not None and "background_edge_hijack" in reason
    assert "ratio" in reason
