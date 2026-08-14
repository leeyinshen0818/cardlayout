from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz
import pytest
from PIL import Image

from cardlayout.models.card_side import CardSide
from cardlayout.models.card_size import MALAYSIA_IC
from cardlayout.services.card_detector import CardDetector
from cardlayout.services.card_processing import CardProcessingService
from cardlayout.services.input_loader import InputLoader


def _scene(
    *,
    canvas: tuple[int, int] = (800, 600),
    center: tuple[int, int] = (400, 300),
    size: tuple[int, int] = (320, 202),
    angle: float = 0.0,
    background: tuple[int, int, int] = (48, 60, 72),
    clutter: bool = False,
) -> Image.Image:
    width, height = canvas
    pixels = np.full((height, width, 3), background, dtype=np.uint8)
    if clutter:
        cv2.rectangle(pixels, (25, 30), (245, 180), (120, 78, 42), -1)
        cv2.circle(pixels, (690, 120), 75, (75, 110, 85), -1)
        cv2.rectangle(pixels, (535, 390), (770, 560), (80, 100, 120), -1)
    box = cv2.boxPoints((center, size, angle)).astype(np.int32)
    cv2.fillConvexPoly(pixels, box, (225, 235, 245))
    cv2.polylines(pixels, [box], True, (10, 10, 10), 5)
    cv2.circle(pixels, center, max(8, size[1] // 9), (65, 105, 185), -1)
    return Image.fromarray(pixels)


@pytest.fixture
def detector() -> CardDetector:
    return CardDetector(MALAYSIA_IC)


def test_clean_rectangular_card_detects_with_high_confidence(detector) -> None:
    result = detector.detect(_scene())
    assert result.success
    assert result.confidence_level == "high"
    assert result.cropped_image is not None
    assert result.cropped_image.width > result.cropped_image.height
    assert result.method.startswith("hybrid:")
    assert result.debug_info["candidate_count"] >= 1


@pytest.mark.parametrize("angle", [12.0, 15.0, 30.0, 35.0, 45.0, 75.0, 90.0, -28.0])
def test_rotated_card_is_made_horizontal(detector, angle: float) -> None:
    result = detector.detect(_scene(angle=angle))
    assert result.success
    assert result.cropped_image is not None
    assert result.cropped_image.width > result.cropped_image.height
    assert abs(result.rotation_angle) == pytest.approx(abs(angle), abs=2.0)


def test_card_with_large_background_uses_full_resolution_crop(detector) -> None:
    image = _scene(canvas=(2400, 1800), center=(1720, 610), size=(620, 391), angle=18)
    result = detector.detect(image)
    assert result.success
    assert result.cropped_image is not None
    assert result.debug_info["working_size"][0] <= 1400
    assert result.cropped_image.width > 600
    assert result.bounding_box is not None


def test_card_near_edge_has_safe_geometry(detector) -> None:
    image = _scene(center=(175, 125), angle=14)
    result = detector.detect(image)
    assert result.success
    assert result.bounding_box is not None
    left, top, right, bottom = result.bounding_box
    assert 0 <= left < right <= image.width
    assert 0 <= top < bottom <= image.height
    assert all(x >= 0 and y >= 0 for x, y in result.polygon_points)


def test_cluttered_background_does_not_default_to_largest_shape(detector) -> None:
    result = detector.detect(_scene(angle=-25, clutter=True))
    assert result.success
    assert result.bounding_box is not None
    left, top, right, bottom = result.bounding_box
    assert right - left < 500
    assert bottom - top < 450
    selected = result.debug_info["selected_candidate"]
    assert set(selected) >= {
        "area_score",
        "shape_score",
        "ratio_score",
        "rectangularity_score",
        "edge_score",
    }


def test_no_card_image_fails_safely(detector) -> None:
    image = Image.new("RGB", (800, 600), (105, 105, 105))
    result = detector.detect(image)
    assert not result.success
    assert result.cropped_image is None
    assert result.confidence_level == "none"


def test_multiple_similar_rectangles_lower_confidence(detector) -> None:
    pixels = np.full((700, 1000, 3), (50, 60, 70), dtype=np.uint8)
    for center in ((260, 250), (740, 450)):
        box = cv2.boxPoints((center, (260, 164), 0)).astype(np.int32)
        cv2.fillConvexPoly(pixels, box, (225, 235, 245))
        cv2.polylines(pixels, [box], True, (10, 10, 10), 5)
    result = detector.detect(Image.fromarray(pixels))
    assert result.success
    assert result.confidence_level == "medium"
    assert result.debug_info["ambiguity_penalty"] >= 0.18


def test_very_small_candidate_is_not_applied(detector) -> None:
    result = detector.detect(_scene(size=(70, 44)))
    assert not result.success
    assert result.confidence_level in {"low", "none"}
    assert result.cropped_image is None


def test_entire_image_rectangle_is_rejected(detector) -> None:
    pixels = np.full((540, 856, 3), 220, dtype=np.uint8)
    cv2.rectangle(pixels, (2, 2), (853, 537), (10, 10, 10), 5)
    result = detector.detect(Image.fromarray(pixels))
    assert not result.success
    assert result.cropped_image is None


def test_already_cropped_detailed_card_keeps_full_image(detector) -> None:
    pixels = np.full((540, 856, 3), (185, 215, 230), dtype=np.uint8)
    for x in range(50, 800, 55):
        cv2.line(pixels, (x, 80), (x - 25, 470), (80, 130, 175), 3)
    cv2.rectangle(pixels, (90, 110), (300, 360), (235, 240, 245), -1)
    cv2.circle(pixels, (195, 220), 65, (75, 105, 155), -1)
    result = detector.detect(Image.fromarray(pixels))
    assert result.success
    assert result.method == "already_cropped"
    assert result.bounding_box == (0, 0, 856, 540)
    assert result.cropped_image is not None
    assert result.cropped_image.size == (856, 540)


def _card_side(image: Image.Image) -> CardSide:
    return CardSide(
        side="front",
        source_path=Path("synthetic.png"),
        source_type="image",
        source_page=None,
        original_image=image.copy(),
        processed_image=image.copy(),
    )


def test_processing_preserves_original_and_reset_restores_it(detector) -> None:
    image = _scene(angle=20, clutter=True)
    side = _card_side(image)
    original_bytes = side.original_image.tobytes()
    service = CardProcessingService(detector)
    result = service.detect(side)
    assert result.success
    assert side.detected_image is result.cropped_image
    assert side.automatic_perspective_result is not None
    assert side.automatic_perspective_result.success
    assert side.processed_image is side.rectified_image
    assert side.original_image.tobytes() == original_bytes

    service.reset(side)
    assert side.detected_image is None
    assert side.detection_result is None
    assert side.processed_image.size == image.size
    assert side.processed_image.tobytes() == original_bytes


def test_failed_detection_does_not_destroy_original(detector) -> None:
    image = Image.new("RGB", (600, 400), (90, 90, 90))
    side = _card_side(image)
    result = CardProcessingService(detector).detect(side)
    assert not result.success
    assert side.detected_image is None
    assert side.processed_image.tobytes() == side.original_image.tobytes()


def test_pdf_rendered_page_uses_same_detector(tmp_path: Path, detector) -> None:
    scene = _scene(angle=16, clutter=True)
    stream = BytesIO()
    scene.save(stream, format="PNG")
    pdf_path = tmp_path / "photo-page.pdf"
    document = fitz.open()
    page = document.new_page(width=800, height=600)
    page.insert_image(page.rect, stream=stream.getvalue())
    document.save(pdf_path)
    document.close()

    side = InputLoader().load_side(pdf_path, "front").card_side
    result = CardProcessingService(detector).detect(side)
    assert side.source_type == "pdf"
    assert result.success
    assert side.detected_image is not None


def test_debug_mode_keeps_intermediate_images_in_memory_only(tmp_path: Path) -> None:
    result = CardDetector(MALAYSIA_IC, debug=True).detect(_scene(angle=10))
    assert result.success
    stages = result.debug_info["stage_images"]
    assert set(stages) >= {
        "grayscale",
        "edges",
        "otsu",
        "adaptive",
        "candidate_overlay",
    }
    assert all(isinstance(image, Image.Image) for image in stages.values())
    assert list(tmp_path.iterdir()) == []
