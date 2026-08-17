from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from cardlayout.models.card_size import MALAYSIA_IC
from cardlayout.services.card_detector import CardDetector


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
    union = (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - intersection
    )
    return intersection / max(1.0, float(union))


def _scene(
    *,
    fill: tuple[int, int, int] = (75, 125, 180),
    background: tuple[int, int, int] = (55, 63, 72),
    quad: np.ndarray | None = None,
    clutter: bool = False,
    multicolor: bool = False,
    finger: bool = False,
) -> tuple[Image.Image, np.ndarray]:
    pixels = np.full((760, 1120, 3), background, dtype=np.uint8)
    if clutter:
        # A monitor and keyboard are deliberately larger and have stronger edges.
        cv2.rectangle(pixels, (45, 42), (745, 460), (32, 36, 42), -1)
        cv2.rectangle(pixels, (45, 42), (745, 460), (8, 10, 12), 7)
        cv2.rectangle(pixels, (70, 67), (720, 420), (77, 82, 88), -1)
        for row in range(575, 725, 24):
            for column in range(40, 690, 45):
                cv2.rectangle(
                    pixels,
                    (column, row),
                    (column + 34, row + 14),
                    (115, 118, 122),
                    1,
                )
    if quad is None:
        quad = np.asarray(((650, 330), (1000, 285), (1025, 505), (675, 550)), np.int32)
    else:
        quad = np.asarray(quad, dtype=np.int32)
    mask = np.zeros(pixels.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad, 255)
    cv2.fillConvexPoly(pixels, quad, fill)
    if multicolor:
        left, top, right, bottom = _bounds(quad)
        band_width = max(1, (right - left) // 4)
        colors = ((195, 35, 45), (245, 205, 45), (35, 155, 95), (45, 80, 185))
        overlay = pixels.copy()
        for index, color in enumerate(colors):
            cv2.rectangle(
                overlay,
                (left + index * band_width, top),
                (right if index == 3 else left + (index + 1) * band_width, bottom),
                color,
                -1,
            )
        pixels[mask > 0] = overlay[mask > 0]
    card_luma = float(np.mean(fill))
    background_luma = float(np.mean(background))
    edge = (18, 18, 18) if max(card_luma, background_luma) > 120 else (225, 225, 225)
    cv2.polylines(pixels, [quad], True, edge, 4, cv2.LINE_AA)

    center = np.round(np.mean(quad, axis=0)).astype(int)
    cv2.circle(pixels, tuple(center), 24, (145, 155, 165), -1, cv2.LINE_AA)
    cv2.line(
        pixels,
        tuple(np.round(quad[3] * 0.68 + quad[0] * 0.32).astype(int)),
        tuple(np.round(quad[2] * 0.68 + quad[1] * 0.32).astype(int)),
        (90, 95, 100),
        3,
        cv2.LINE_AA,
    )
    if finger:
        corner = quad[1]
        adjacent = np.round(quad[1] * 0.48 + quad[2] * 0.52).astype(int)
        cv2.line(
            pixels,
            tuple(corner),
            tuple(adjacent),
            (190, 143, 112),
            28,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            pixels,
            tuple(corner),
            (55, 38),
            -10,
            0,
            360,
            (190, 143, 112),
            -1,
            cv2.LINE_AA,
        )
    return Image.fromarray(pixels), quad


def _assert_detects_physical_card(
    image: Image.Image, expected: np.ndarray, *, minimum_iou: float = 0.72
) -> None:
    result = CardDetector(MALAYSIA_IC).detect(image)
    assert result.success, result.debug_info.get("confidence_reasoning")
    assert result.bounding_box is not None
    assert _iou(result.bounding_box, _bounds(expected)) >= minimum_iou
    points = np.asarray(result.polygon_points, dtype=np.float32)
    assert points.shape == (4, 2)
    assert cv2.isContourConvex(np.round(points).astype(np.int32))
    tl, tr, br, bl = points
    assert tl[0] < tr[0] and bl[0] < br[0]
    assert tl[1] < bl[1] and tr[1] < br[1]


@pytest.mark.parametrize(
    ("name", "fill", "background", "multicolor"),
    [
        ("blue_ic", (65, 115, 175), (56, 62, 70), False),
        ("white_grey_licence", (224, 226, 225), (72, 78, 84), False),
        ("dark_card", (28, 32, 38), (205, 207, 210), False),
        ("red_card", (178, 42, 48), (62, 67, 72), False),
        ("low_saturation_card", (148, 153, 151), (48, 53, 58), False),
        ("multicolored_membership", (120, 120, 120), (55, 60, 66), True),
    ],
)
def test_card_color_is_not_a_detection_requirement(
    name: str,
    fill: tuple[int, int, int],
    background: tuple[int, int, int],
    multicolor: bool,
) -> None:
    del name
    image, card = _scene(fill=fill, background=background, multicolor=multicolor)
    _assert_detects_physical_card(image, card)


@pytest.mark.parametrize(
    ("background", "fill"),
    [((24, 28, 32), (210, 214, 218)), ((235, 237, 239), (55, 62, 68))],
)
def test_card_detects_on_dark_and_light_backgrounds(
    background: tuple[int, int, int], fill: tuple[int, int, int]
) -> None:
    image, card = _scene(background=background, fill=fill)
    _assert_detects_physical_card(image, card)


def test_distant_card_and_reasonable_non_ic_ratio_remain_credible() -> None:
    card = cv2.boxPoints(((830, 470), (175, 118), -24)).astype(np.int32)
    image, card = _scene(fill=(210, 214, 220), quad=card)
    _assert_detects_physical_card(image, card, minimum_iou=0.67)


def test_strong_perspective_quad_is_detected_and_corner_order_is_normalized() -> None:
    card = np.asarray(((250, 215), (910, 125), (810, 585), (330, 625)), np.int32)
    image, card = _scene(fill=(205, 208, 212), quad=card)
    _assert_detects_physical_card(image, card, minimum_iou=0.78)


def test_finger_occlusion_recovers_physical_perimeter_for_review() -> None:
    image, card = _scene(fill=(78, 126, 168), finger=True)
    _assert_detects_physical_card(image, card, minimum_iou=0.58)


def test_card_beats_larger_monitor_and_keyboard_rectangles() -> None:
    image, card = _scene(
        fill=(215, 217, 220),
        quad=np.asarray(((770, 470), (1050, 425), (1065, 605), (785, 650))),
        clutter=True,
    )
    _assert_detects_physical_card(image, card, minimum_iou=0.70)


def test_card_near_image_edge_keeps_safe_physical_corners() -> None:
    card = np.asarray(((8, 190), (345, 145), (365, 365), (25, 408)), np.int32)
    image, card = _scene(fill=(182, 45, 55), quad=card)
    _assert_detects_physical_card(image, card, minimum_iou=0.72)
