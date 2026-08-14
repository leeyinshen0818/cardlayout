from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from cardlayout.models.card_side import CardSide
from cardlayout.models.card_size import CardSizePreset, MALAYSIA_IC
from cardlayout.models.detection import CardDetectionResult
from cardlayout.services.card_processing import CardProcessingService
from cardlayout.services.perspective_corrector import PerspectiveCorrector


def _photo(size: tuple[int, int] = (900, 650)) -> Image.Image:
    image = Image.new("RGB", size, (37, 45, 58))
    draw = ImageDraw.Draw(image)
    for x in range(0, size[0], 20):
        draw.line((x, 0, x, size[1]), fill=(70 + x % 90, 90, 120), width=2)
    for y in range(0, size[1], 20):
        draw.line((0, y, size[0], y), fill=(80, 70 + y % 100, 110), width=2)
    return image


@pytest.mark.parametrize(
    "points",
    [
        ((120, 100), (760, 100), (760, 505), (120, 505)),
        ((150, 120), (735, 145), (790, 500), (105, 520)),
        ((260, 90), (690, 190), (835, 530), (65, 570)),
    ],
)
def test_rectangles_and_mild_or_severe_trapezoids_are_rectified(points) -> None:
    result = PerspectiveCorrector(MALAYSIA_IC).correct(
        _photo(), points, detector_confidence=0.9
    )

    assert result.success
    assert result.rectified_image is not None
    assert result.rectified_image.width > result.rectified_image.height
    assert result.rectified_image.width / result.rectified_image.height == pytest.approx(
        85.6 / 54.0, abs=0.002
    )
    assert len(result.transform_matrix) == 3


def test_rotated_trapezoid_and_arbitrary_input_order() -> None:
    corrector = PerspectiveCorrector(MALAYSIA_IC)
    rotated = cv2.boxPoints(((440, 330), (560, 353), 36)).astype(np.float32)
    arbitrary = tuple(tuple(map(float, rotated[index])) for index in (2, 0, 3, 1))

    ordered = corrector.order_corners(arbitrary)
    valid, warning = corrector.validate_quad(ordered, (900, 650))
    result = corrector.correct(_photo(), arbitrary, detector_confidence=0.95)

    assert valid, warning
    assert result.success
    assert result.output_dimensions is not None
    assert result.output_dimensions[0] > result.output_dimensions[1]
    assert {tuple(np.round(point, 3)) for point in ordered} == {
        tuple(np.round(point, 3)) for point in rotated
    }


def test_corner_order_preserves_content_mapping_through_homography() -> None:
    card = np.zeros((540, 856, 3), dtype=np.uint8)
    card[:270, :428] = (230, 30, 30)
    card[:270, 428:] = (30, 220, 40)
    card[270:, 428:] = (35, 65, 230)
    card[270:, :428] = (235, 210, 25)
    source = np.asarray(((0, 0), (855, 0), (855, 539), (0, 539)), np.float32)
    trapezoid = np.asarray(((150, 100), (760, 150), (830, 560), (85, 520)), np.float32)
    matrix = cv2.getPerspectiveTransform(source, trapezoid)
    photo = cv2.warpPerspective(card, matrix, (900, 650))

    result = PerspectiveCorrector(MALAYSIA_IC).correct(
        Image.fromarray(photo),
        tuple(tuple(map(float, trapezoid[index])) for index in (2, 0, 3, 1)),
    )
    assert result.success and result.rectified_image is not None
    pixels = np.asarray(result.rectified_image)
    height, width = pixels.shape[:2]
    samples = (
        pixels[height // 4, width // 4],
        pixels[height // 4, width * 3 // 4],
        pixels[height * 3 // 4, width * 3 // 4],
        pixels[height * 3 // 4, width // 4],
    )
    assert samples[0][0] > 180
    assert samples[1][1] > 180
    assert samples[2][2] > 180
    assert samples[3][0] > 180 and samples[3][1] > 160


def test_invalid_crossing_duplicate_and_near_zero_quads_are_rejected() -> None:
    corrector = PerspectiveCorrector(MALAYSIA_IC)
    crossing = ((10, 10), (100, 100), (100, 10), (10, 100))
    valid, warning = corrector.validate_quad(crossing, (200, 200))
    assert not valid and warning == "Corner lines cannot cross."

    duplicate = corrector.correct(
        _photo(), ((10, 10), (100, 10), (100, 10), (10, 100))
    )
    assert not duplicate.success
    assert "overlap" in (duplicate.warning or "").lower()

    tiny = corrector.correct(
        _photo(), ((10, 10), (12, 10), (12, 11), (10, 11))
    )
    assert not tiny.success


def test_selected_card_preset_controls_exact_output_ratio() -> None:
    preset = CardSizePreset("Wide test card", 100.0, 50.0)
    result = PerspectiveCorrector(preset).correct(
        _photo(), ((100, 100), (800, 145), (760, 500), (120, 520))
    )
    assert result.success and result.rectified_image is not None
    assert result.rectified_image.width / result.rectified_image.height == pytest.approx(
        2.0, abs=0.001
    )


def test_output_policy_avoids_large_upscales_and_caps_large_sources() -> None:
    corrector = PerspectiveCorrector(MALAYSIA_IC)
    moderate = corrector.correct(
        _photo((1000, 700)),
        ((100, 100), (800, 100), (800, 541), (100, 541)),
        refine=False,
    )
    large = corrector.correct(
        _photo((1900, 1300)),
        ((100, 100), (1700, 100), (1700, 1110), (100, 1110)),
        refine=False,
    )
    assert moderate.output_dimensions is not None
    assert moderate.output_dimensions[0] <= 700 * 1.15 + 2
    assert large.output_dimensions is not None
    assert large.output_dimensions[0] <= 1200


def test_debug_mode_reports_geometry_without_storing_images() -> None:
    result = PerspectiveCorrector(MALAYSIA_IC, debug=True).correct(
        _photo(), ((100, 100), (800, 120), (780, 520), (120, 510))
    )
    assert result.success
    assert "source_corner_points" in result.debug_info
    assert "refined_corner_points" in result.debug_info
    assert "destination_rectangle" in result.debug_info
    assert "transform_matrix" in result.debug_info
    assert not any(isinstance(value, Image.Image) for value in result.debug_info.values())


def test_out_of_bounds_points_are_clamped_and_flagged_for_review() -> None:
    result = PerspectiveCorrector(MALAYSIA_IC).correct(
        _photo(), ((-40, 80), (850, 90), (940, 590), (-35, 620)), detector_confidence=0.9
    )
    assert result.success
    assert result.status == "review"
    assert "outside" in (result.warning or "").lower()
    assert all(0 <= x < 900 and 0 <= y < 650 for x, y in result.source_points)


def test_medium_detector_confidence_requires_correction_review() -> None:
    result = PerspectiveCorrector(MALAYSIA_IC).correct(
        _photo(),
        ((100, 100), (800, 100), (800, 540), (100, 540)),
        detector_confidence=0.65,
    )
    assert result.success
    assert result.status == "review"
    assert result.status_text == "Detected · Review correction"


def _side(image: Image.Image) -> CardSide:
    return CardSide(
        side="front",
        source_path=Path("local-test.png"),
        source_type="image",
        source_page=None,
        original_image=image.copy(),
        processed_image=image.copy(),
    )


def test_original_detected_and_corrected_stages_remain_separate() -> None:
    original = _photo()
    original_bytes = original.tobytes()
    side = _side(original)
    detected = original.crop((100, 100, 800, 540))
    detection = CardDetectionResult(
        success=True,
        confidence=0.9,
        confidence_level="high",
        polygon_points=((100, 100), (800, 120), (780, 540), (120, 520)),
        cropped_image=detected,
    )
    side.apply_detection(detection)
    correction = PerspectiveCorrector(MALAYSIA_IC).correct(
        side.original_image, detection.polygon_points, detector_confidence=0.9
    )
    side.apply_automatic_correction(correction)

    assert side.original_image.tobytes() == original_bytes
    assert side.detected_image is detected
    assert side.rectified_image is correction.rectified_image
    assert side.best_image is correction.rectified_image
    assert side.processed_image is side.best_image


def test_manual_correction_overrides_automatic_and_reset_restores_it() -> None:
    original = _photo()
    side = _side(original)
    auto = PerspectiveCorrector(MALAYSIA_IC).correct(
        original, ((100, 100), (800, 110), (780, 530), (120, 520))
    )
    manual = PerspectiveCorrector(MALAYSIA_IC).correct(
        original,
        ((130, 120), (770, 130), (750, 500), (145, 490)),
        method="manual",
        refine=False,
    )
    side.apply_automatic_correction(auto)
    side.apply_manual_correction(manual)

    assert side.has_manual_correction
    assert side.best_image is manual.rectified_image
    assert side.correction_status_text == "Manual correction applied"

    side.reset_correction()
    assert not side.has_manual_correction
    assert side.best_image is auto.rectified_image


class _FixedDetector:
    card_size = MALAYSIA_IC

    def __init__(self, result: CardDetectionResult) -> None:
        self.result = result

    def detect(self, image: Image.Image) -> CardDetectionResult:
        del image
        return self.result


def test_processing_service_automatically_corrects_and_redetection_invalidates_manual() -> None:
    image = _photo()
    detection = CardDetectionResult(
        success=True,
        confidence=0.88,
        confidence_level="high",
        polygon_points=((120, 100), (780, 140), (750, 530), (105, 500)),
        cropped_image=image.crop((100, 90, 790, 540)),
        debug_info={"inferred_edge_count": 1},
    )
    side = _side(image)
    service = CardProcessingService(_FixedDetector(detection))  # type: ignore[arg-type]
    service.detect(side)

    assert side.automatic_perspective_result is not None
    assert side.automatic_perspective_result.success
    assert side.automatic_perspective_result.status == "review"
    manual = service.apply_manual_correction(side, detection.polygon_points)
    assert manual.success and side.has_manual_correction

    service.detect(side)
    assert not side.has_manual_correction
    assert side.automatic_perspective_result is not None


def test_reset_detection_preserves_original_but_clears_all_later_stages() -> None:
    image = _photo()
    side = _side(image)
    original_bytes = side.original_image.tobytes()
    correction = PerspectiveCorrector(MALAYSIA_IC).correct(
        image, ((100, 100), (800, 100), (800, 540), (100, 540))
    )
    side.apply_automatic_correction(correction)
    side.reset_detection()

    assert side.original_image.tobytes() == original_bytes
    assert side.detected_image is None
    assert side.rectified_image is None
    assert side.active_perspective_result is None
    assert side.best_image.tobytes() == original_bytes
