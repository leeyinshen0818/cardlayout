from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from cardlayout.models.card_size import MALAYSIA_IC
from cardlayout.models.detection_config import CardDetectionConfig
from cardlayout.services.card_detector import CardDetector


def _background(width: int, height: int, clutter: bool = True) -> np.ndarray:
    pixels = np.full((height, width, 3), (72, 78, 82), dtype=np.uint8)
    if clutter:
        cv2.rectangle(pixels, (30, 40), (460, 260), (115, 105, 90), -1)
        cv2.rectangle(pixels, (30, 40), (460, 260), (25, 25, 25), 5)
        screen_left = max(500, width - 800)
        cv2.rectangle(
            pixels,
            (screen_left, 40),
            (min(width - 40, screen_left + 360), 280),
            (55, 58, 65),
            -1,
        )
        cv2.rectangle(
            pixels,
            (screen_left, 40),
            (min(width - 40, screen_left + 360), 280),
            (20, 20, 20),
            6,
        )
        for y in range(int(height * 0.75), height - 20, 25):
            cv2.line(pixels, (40, y), (min(650, width - 20), y), (110, 110, 110), 2)
    return pixels


def _draw_card(
    pixels: np.ndarray,
    center: tuple[int, int],
    size: tuple[int, int],
    angle: float,
    fill: tuple[int, int, int] = (220, 230, 240),
    edge: tuple[int, int, int] = (10, 10, 10),
) -> np.ndarray:
    box = cv2.boxPoints((center, size, angle)).astype(np.int32)
    cv2.fillConvexPoly(pixels, box, fill)
    cv2.polylines(pixels, [box], True, edge, 5)
    cv2.circle(
        pixels,
        center,
        max(8, size[1] // 10),
        (60, 100, 180),
        -1,
    )
    cv2.line(
        pixels,
        (center[0] - size[0] // 5, center[1] + size[1] // 5),
        (center[0] + size[0] // 4, center[1] + size[1] // 5),
        (80, 115, 155),
        3,
    )
    return box


def _bounds(points: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(points[:, 0].min()),
        int(points[:, 1].min()),
        int(points[:, 0].max()),
        int(points[:, 1].max()),
    )


def _iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / float(first_area + second_area - intersection)


def test_far_away_card_beats_larger_background_rectangles() -> None:
    pixels = _background(1600, 1000)
    card_box = _draw_card(pixels, (1150, 620), (190, 120), 27)
    result = CardDetector(MALAYSIA_IC).detect(Image.fromarray(pixels))
    assert result.success
    assert result.confidence_level in {"high", "medium"}
    assert result.bounding_box is not None
    assert _iou(result.bounding_box, _bounds(card_box)) > 0.72
    assert len(result.debug_info["passes"]) >= 2
    assert result.debug_info["passes"][0]["scale_used"] <= 1100
    assert max(result.debug_info["scale_used"]) > 1100


def test_card_around_ten_percent_of_frame_detects_in_clutter() -> None:
    pixels = _background(1400, 900)
    card_box = _draw_card(pixels, (950, 570), (430, 271), -16)
    result = CardDetector(MALAYSIA_IC).detect(Image.fromarray(pixels))
    assert result.success
    assert result.bounding_box is not None
    assert _iou(result.bounding_box, _bounds(card_box)) > 0.78


@pytest.mark.parametrize("occlusion", ["edge", "corner", "two_adjacent"])
def test_finger_like_occlusion_returns_correct_card_for_review(
    occlusion: str,
) -> None:
    pixels = _background(1200, 800)
    card_box = _draw_card(pixels, (720, 470), (360, 227), 20)
    finger = (185, 145, 115)
    if occlusion == "edge":
        midpoint = ((card_box[0] + card_box[1]) // 2).astype(int)
        cv2.ellipse(pixels, tuple(midpoint), (95, 50), 20, 0, 360, finger, -1)
    elif occlusion == "corner":
        cv2.ellipse(pixels, tuple(card_box[0]), (72, 48), 15, 0, 360, finger, -1)
    else:
        first_end = (card_box[0] * 0.35 + card_box[1] * 0.65).astype(int)
        second_end = (card_box[1] * 0.40 + card_box[2] * 0.60).astype(int)
        cv2.line(pixels, tuple(card_box[0]), tuple(first_end), finger, 18)
        cv2.line(pixels, tuple(card_box[1]), tuple(second_end), finger, 18)

    result = CardDetector(MALAYSIA_IC).detect(Image.fromarray(pixels))
    assert result.success
    assert result.confidence_level in {"medium", "high"}
    assert result.bounding_box is not None
    assert _iou(result.bounding_box, _bounds(card_box)) > 0.55


def test_three_visible_edges_infer_the_fourth() -> None:
    pixels = np.full((700, 1000, 3), (100, 105, 110), dtype=np.uint8)
    card_box = cv2.boxPoints(((560, 380), (330, 208), 28)).astype(np.int32)
    for index in (0, 1, 2):
        cv2.line(
            pixels,
            tuple(card_box[index]),
            tuple(card_box[(index + 1) % 4]),
            (15, 15, 15),
            5,
        )
    cv2.circle(pixels, (560, 380), 18, (50, 65, 90), -1)
    result = CardDetector(MALAYSIA_IC).detect(Image.fromarray(pixels))
    assert result.success
    assert result.confidence_level == "medium"
    assert result.bounding_box is not None
    assert _iou(result.bounding_box, _bounds(card_box)) > 0.72
    selected = result.debug_info["selected_candidate"]
    assert selected["inferred_edges"] == 1
    assert any(method.startswith("partial3") for method in selected["source_methods"])


def test_two_opposite_edges_construct_review_candidate() -> None:
    pixels = np.full((700, 1000, 3), (100, 105, 110), dtype=np.uint8)
    card_box = cv2.boxPoints(((560, 380), (330, 208), 28)).astype(np.int32)
    for index in (0, 2):
        cv2.line(
            pixels,
            tuple(card_box[index]),
            tuple(card_box[(index + 1) % 4]),
            (15, 15, 15),
            5,
        )
    cv2.circle(pixels, (560, 380), 18, (50, 65, 90), -1)
    result = CardDetector(MALAYSIA_IC).detect(Image.fromarray(pixels))
    assert result.success
    assert result.confidence_level == "medium"
    assert result.bounding_box is not None
    assert _iou(result.bounding_box, _bounds(card_box)) > 0.72
    assert result.debug_info["selected_candidate"]["inferred_edges"] == 2


def test_color_edges_find_low_grayscale_contrast_card() -> None:
    pixels = np.full((800, 1200, 3), (105, 118, 102), dtype=np.uint8)
    card_box = _draw_card(
        pixels,
        (700, 430),
        (350, 221),
        -18,
        fill=(90, 118, 130),
        edge=(95, 108, 100),
    )
    result = CardDetector(MALAYSIA_IC).detect(Image.fromarray(pixels))
    assert result.success
    assert result.bounding_box is not None
    assert _iou(result.bounding_box, _bounds(card_box)) > 0.75
    assert any(
        "detailed_hybrid" == detection_pass["name"]
        for detection_pass in result.debug_info["passes"]
    )


def test_debug_reasoning_contains_lines_scales_scores_and_timing() -> None:
    pixels = _background(1200, 800)
    _draw_card(pixels, (720, 470), (330, 208), 30)
    result = CardDetector(MALAYSIA_IC, debug=True).detect(Image.fromarray(pixels))
    assert result.success
    debug = result.debug_info
    assert debug["candidate_count"] >= 1
    assert debug["best_score"] is not None
    assert "second_best_score" in debug
    assert {
        "selected_candidate_score",
        "second_candidate_score",
        "full_card_boundary_score",
        "coverage_score",
        "ratio_score",
        "internal_subregion_penalty",
        "background_penalty",
    } <= debug.keys()
    assert debug["processing_time_ms"] > 0
    assert debug["scale_used"]
    assert "inferred_edge_count" in debug
    assert any(item["line_segment_count"] > 0 for item in debug["passes"])
    assert set(debug["stage_images"]) >= {
        "original",
        "candidate_scores",
        "selected_rough_card_region",
        "detailed_hybrid_lines",
        "detailed_hybrid_edges",
    }


def test_detection_scales_and_thresholds_are_configurable() -> None:
    config = CardDetectionConfig(
        fast_long_edge=700,
        detailed_long_edge=900,
        fallback_long_edge=1000,
    )
    pixels = _background(1200, 800)
    _draw_card(pixels, (720, 470), (300, 189), 12)
    detector = CardDetector(MALAYSIA_IC, config=config)
    result = detector.detect(Image.fromarray(pixels))
    assert detector.config is config
    assert result.success
    assert result.debug_info["passes"][0]["scale_used"] == 700
    assert max(result.debug_info["scale_used"]) <= 1000
