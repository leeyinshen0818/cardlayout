from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz
import pytest
from PIL import Image

from cardlayout.models.card_size import MALAYSIA_IC
from cardlayout.services.card_detector import CardDetector
from cardlayout.services.card_processing import CardProcessingService
from cardlayout.services.input_loader import InputLoader
from cardlayout.services.pdf_loader import PDFLoader
from cardlayout.services.perspective_corrector import PerspectiveCorrector


def _card_raster(side: str = "front") -> Image.Image:
    pixels = np.full((850, 1200, 3), (63, 76, 91), dtype=np.uint8)
    card = np.asarray(((205, 205), (995, 175), (1010, 675), (190, 705)), np.int32)
    cv2.fillConvexPoly(pixels, card, (112, 166, 207))
    cv2.polylines(pixels, [card], True, (15, 22, 30), 7, cv2.LINE_AA)
    if side == "front":
        cv2.circle(pixels, (815, 390), 92, (42, 70, 105), -1, cv2.LINE_AA)
        for y, length in ((310, 280), (350, 350), (390, 235), (535, 430)):
            cv2.line(pixels, (285, y), (285 + length, y), (45, 83, 122), 9)
    else:
        cv2.circle(pixels, (285, 270), 17, (86, 141, 192), -1, cv2.LINE_AA)
    return Image.fromarray(pixels)


def _image_bytes(image: Image.Image, image_format: str = "PNG") -> bytes:
    output = BytesIO()
    image.save(output, format=image_format, quality=96)
    return output.getvalue()


def _write_image(path: Path, image: Image.Image, image_format: str = "PNG") -> Path:
    image.save(path, format=image_format, quality=96)
    return path


def _write_pdf(
    path: Path,
    images: list[Image.Image],
    *,
    white_margin: bool = False,
    rotation: int = 0,
    page_size: tuple[float, float] = (600.0, 425.0),
    inset_fraction: float = 0.15,
    page_background: tuple[float, float, float] | None = None,
) -> Path:
    document = fitz.open()
    for image in images:
        page = document.new_page(width=page_size[0], height=page_size[1])
        if page_background is not None:
            page.draw_rect(page.rect, fill=page_background, color=page_background)
        if white_margin:
            inset_x = page.rect.width * inset_fraction
            inset_y = page.rect.height * inset_fraction
            target = fitz.Rect(
                inset_x,
                inset_y,
                page.rect.width - inset_x,
                page.rect.height - inset_y,
            )
        else:
            target = page.rect
        page.insert_image(target, stream=_image_bytes(image))
        if rotation:
            page.set_rotation(rotation)
    document.save(path)
    document.close()
    return path


def _rounded_card_asset(side: str) -> Image.Image:
    width, height = 900, 568
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (32, 0), (width - 33, height - 1), 255, -1)
    cv2.rectangle(mask, (0, 32), (width - 1, height - 33), 255, -1)
    for center in ((32, 32), (width - 33, 32), (width - 33, height - 33), (32, height - 33)):
        cv2.circle(mask, center, 32, 255, -1, cv2.LINE_AA)
    rgba[:, :, :3] = (92, 153, 202)
    rgba[:, :, 3] = mask
    if side == "front":
        cv2.rectangle(rgba, (70, 75), (310, 185), (250, 250, 247, 255), -1)
        cv2.circle(rgba, (720, 275), 105, (45, 74, 110, 255), -1, cv2.LINE_AA)
        for y in (265, 315, 365, 455):
            cv2.line(rgba, (95, y), (570, y), (42, 82, 124, 255), 10)
    else:
        cv2.circle(rgba, (110, 480), 18, (112, 174, 216, 255), -1, cv2.LINE_AA)
    return Image.fromarray(rgba, "RGBA")


def _normalized_polygon(result, size: tuple[int, int]) -> np.ndarray:
    points = PerspectiveCorrector.order_corners(result.polygon_points).astype(np.float64)
    points[:, 0] /= size[0]
    points[:, 1] /= size[1]
    return points


def test_pdf_and_png_use_canonical_rgb_normalization(tmp_path: Path) -> None:
    image = _card_raster()
    png = _write_image(tmp_path / "card.png", image)
    pdf = _write_pdf(tmp_path / "card.pdf", [image])
    loader = InputLoader(debug=True)

    image_side = loader.load_side(png, "front").card_side
    pdf_side = loader.load_side(pdf, "front").card_side

    assert image_side.processing_raster.mode == "RGB"
    assert pdf_side.processing_raster.mode == "RGB"
    assert image_side.detector_input_image is not None
    assert pdf_side.detector_input_image is not None
    assert pdf_side.original_pdf_render is not None
    assert pdf_side.normalized_pdf_raster is not None
    assert {
        "raw_pdf_render",
        "detected_outer_background",
        "trim_bounding_box",
        "trimmed_pdf_raster",
        "normalized_pdf_raster",
        "detector_input",
        "final_card_detector_input",
    } <= set(pdf_side.source_diagnostics["stage_images"])


def test_pdf_render_and_exported_raster_have_exact_ab_detector_parity(
    tmp_path: Path,
) -> None:
    pdf = _write_pdf(tmp_path / "ab.pdf", [_card_raster()])
    loader = InputLoader()
    pdf_side = loader.load_side(pdf, "front").card_side
    exported = _write_image(
        tmp_path / "pdf-render.png", pdf_side.processing_raster, "PNG"
    )
    image_side = loader.load_side(exported, "front").card_side
    detector = CardDetector(MALAYSIA_IC)

    pdf_result = detector.detect(pdf_side.processing_raster)
    image_result = detector.detect(image_side.processing_raster)

    assert pdf_result.success and image_result.success
    assert pdf_side.processing_raster.size == image_side.processing_raster.size
    assert pdf_result.bounding_box == image_result.bounding_box
    assert np.allclose(pdf_result.polygon_points, image_result.polygon_points)
    assert pdf_result.confidence == pytest.approx(image_result.confidence, abs=1e-8)


def test_equivalent_png_and_pdf_detection_and_perspective_are_geometrically_equal(
    tmp_path: Path,
) -> None:
    image = _card_raster()
    png = _write_image(tmp_path / "front.png", image)
    pdf = _write_pdf(tmp_path / "front.pdf", [image])
    loader = InputLoader()
    sides = [
        loader.load_side(png, "front").card_side,
        loader.load_side(pdf, "front").card_side,
    ]
    processor = CardProcessingService(CardDetector(MALAYSIA_IC))
    results = [processor.detect(side) for side in sides]

    assert all(result.success for result in results)
    polygons = [
        _normalized_polygon(result, side.processing_raster.size)
        for result, side in zip(results, sides)
    ]
    assert np.max(np.abs(polygons[0] - polygons[1])) < 0.012
    assert results[0].confidence == pytest.approx(results[1].confidence, abs=0.08)
    assert all(side.automatic_perspective_result is not None for side in sides)
    corrected_sizes = [side.geometry_image.size for side in sides]
    corrected_ratios = [width / height for width, height in corrected_sizes]
    assert corrected_ratios[0] == pytest.approx(corrected_ratios[1], abs=0.012)


def test_two_page_pdf_preserves_front_back_order_and_detects_low_texture_back(
    tmp_path: Path,
) -> None:
    pdf = _write_pdf(
        tmp_path / "two-page.pdf",
        [_card_raster("front"), _card_raster("back")],
        white_margin=True,
        page_size=(612, 792),
    )
    loaded = InputLoader().load_two_page_pdf(pdf)
    assert loaded.front.source_page == 1
    assert loaded.back is not None and loaded.back.source_page == 2
    processor = CardProcessingService(CardDetector(MALAYSIA_IC))

    front = processor.detect(loaded.front)
    back = processor.detect(loaded.back)

    assert front.success and back.success
    assert loaded.front.source_diagnostics["white_margin_trim_applied"] is True
    assert loaded.back.source_diagnostics["white_margin_trim_applied"] is True
    for side, result in ((loaded.front, front), (loaded.back, back)):
        assert side.source_diagnostics["detector_input_width"] == side.processing_raster.width
        assert result.debug_info["selected_candidate"]["area_ratio"] >= 0.33


def test_separate_pdf_and_mixed_inputs_keep_source_state_independent(
    tmp_path: Path,
) -> None:
    front_image, back_image = _card_raster("front"), _card_raster("front")
    front_pdf = _write_pdf(tmp_path / "front.pdf", [front_image])
    back_pdf = _write_pdf(tmp_path / "back.pdf", [back_image])
    front_jpg = _write_image(tmp_path / "front.jpg", front_image, "JPEG")
    back_png = _write_image(tmp_path / "back.png", back_image)
    loader = InputLoader()

    separate = (
        loader.load_side(front_pdf, "front").card_side,
        loader.load_side(back_pdf, "back").card_side,
    )
    mixed_a = (
        loader.load_side(front_jpg, "front").card_side,
        loader.load_side(back_pdf, "back").card_side,
    )
    mixed_b = (
        loader.load_side(front_pdf, "front").card_side,
        loader.load_side(back_png, "back").card_side,
    )

    assert [side.source_type for side in separate] == ["pdf", "pdf"]
    assert [side.source_type for side in mixed_a] == ["image", "pdf"]
    assert [side.source_type for side in mixed_b] == ["pdf", "image"]
    assert all(
        side.processing_raster.mode == "RGB"
        for pair in (separate, mixed_a, mixed_b)
        for side in pair
    )
    detector = CardDetector(MALAYSIA_IC)
    detections = [
        [detector.detect(side.processing_raster) for side in pair]
        for pair in (separate, mixed_a, mixed_b)
    ]
    assert all(result.success for pair in detections for result in pair)
    for side_index in (0, 1):
        reference = _normalized_polygon(
            detections[0][side_index], separate[side_index].processing_raster.size
        )
        for pair, results in zip((mixed_a, mixed_b), detections[1:]):
            candidate = _normalized_polygon(
                results[side_index], pair[side_index].processing_raster.size
            )
            assert np.max(np.abs(reference - candidate)) < 0.018


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_pdf_page_rotation_is_applied_once_and_recorded(
    tmp_path: Path, rotation: int
) -> None:
    image = Image.new("RGB", (400, 600), (80, 120, 160))
    pdf = _write_pdf(
        tmp_path / f"rotated-{rotation}.pdf",
        [image],
        rotation=rotation,
        page_size=(400, 600),
    )
    side = InputLoader().load_side(pdf, "front").card_side

    assert side.source_diagnostics["pdf_rotation"] == rotation
    width, height = side.original_pdf_render.size  # type: ignore[union-attr]
    assert (width > height) is (rotation in (90, 270))
    assert side.processing_raster.size == (width, height)


def test_white_margin_trim_is_conservative_and_colored_background_is_preserved(
    tmp_path: Path,
) -> None:
    image = _card_raster()
    white_pdf = _write_pdf(
        tmp_path / "white-margin.pdf",
        [image],
        white_margin=True,
        page_size=(612, 792),
    )
    full_pdf = _write_pdf(tmp_path / "full-photo.pdf", [image])
    loader = InputLoader()
    white = loader.load_side(white_pdf, "front").card_side
    full = loader.load_side(full_pdf, "front").card_side

    assert white.source_diagnostics["white_margin_trim_applied"] is True
    assert white.source_diagnostics["trim_box"] is not None
    assert white.processing_raster.width < white.original_pdf_render.width  # type: ignore[union-attr]
    assert full.source_diagnostics["white_margin_trim_applied"] is False
    assert full.processing_raster.size == full.original_pdf_render.size  # type: ignore[union-attr]


def test_render_resolution_adapts_and_remains_capped(tmp_path: Path) -> None:
    image = _card_raster()
    small = _write_pdf(
        tmp_path / "small-page.pdf", [image], page_size=(240, 160)
    )
    large = _write_pdf(
        tmp_path / "large-page.pdf", [image], page_size=(1800, 1200)
    )
    loader = InputLoader(pdf_loader=PDFLoader())
    small_side = loader.load_side(small, "front").card_side
    large_side = loader.load_side(large, "front").card_side

    assert small_side.source_diagnostics["effective_render_dpi"] == 300.0
    assert large_side.source_diagnostics["effective_render_dpi"] <= 252.0
    assert max(large_side.original_pdf_render.size) <= 4200  # type: ignore[union-attr]
    assert small_side.source_diagnostics["render_zoom"] > 240 / 72


def test_tiny_content_region_triggers_one_capped_high_resolution_rerender(
    tmp_path: Path,
) -> None:
    pdf = _write_pdf(
        tmp_path / "tiny-content.pdf",
        [_card_raster()],
        white_margin=True,
        page_size=(612, 792),
        inset_fraction=0.38,
    )
    side = InputLoader().load_side(pdf, "front").card_side

    assert side.source_diagnostics["initial_render_dpi"] == 240.0
    assert side.source_diagnostics["adaptive_rerender_applied"] is True
    assert side.source_diagnostics["effective_render_dpi"] == 300.0
    assert max(side.original_pdf_render.size) <= 4200  # type: ignore[union-attr]


@pytest.mark.parametrize("side_name", ["front", "back"])
@pytest.mark.parametrize(
    "page_background",
    [(1.0, 1.0, 1.0), (0.96, 0.94, 0.90), (0.82, 0.82, 0.82)],
    ids=("white", "cream", "light-gray"),
)
def test_border_connected_pdf_frame_is_removed_before_full_card_detection(
    tmp_path: Path,
    side_name: str,
    page_background: tuple[float, float, float],
) -> None:
    pdf = _write_pdf(
        tmp_path / f"{side_name}-{page_background[0]}.pdf",
        [_rounded_card_asset(side_name)],
        white_margin=True,
        inset_fraction=0.11,
        page_background=page_background,
    )
    side = InputLoader(debug=True).load_side(pdf, side_name).card_side  # type: ignore[arg-type]
    raw_size = side.original_pdf_render.size  # type: ignore[union-attr]

    result = CardProcessingService(CardDetector(MALAYSIA_IC)).detect(side)

    assert side.source_diagnostics["white_margin_trim_applied"] is True
    assert side.source_diagnostics["outer_background_confidence"] >= 0.82
    assert side.processing_raster.width < raw_size[0] * 0.90
    assert side.processing_raster.height < raw_size[1] * 0.90
    assert result.success
    assert result.method == "already_cropped" or (
        result.debug_info["selected_candidate"]["area_ratio"] >= 0.70
    )
    assert side.automatic_perspective_result is not None
    assert side.automatic_perspective_result.success
    refined = np.asarray(side.automatic_perspective_result.refined_points)
    assert float(refined[:, 0].min()) >= 1.5
    assert float(refined[:, 1].min()) >= 1.5
    assert float(refined[:, 0].max()) <= side.processing_raster.width - 1.5
    assert float(refined[:, 1].max()) <= side.processing_raster.height - 1.5
    corrected = np.asarray(side.geometry_image.convert("RGB"))
    border = np.concatenate(
        (
            corrected[:12].reshape(-1, 3),
            corrected[-12:].reshape(-1, 3),
            corrected[12:-12, :12].reshape(-1, 3),
            corrected[12:-12, -12:].reshape(-1, 3),
        )
    )
    assert float(np.mean(np.all(border >= 238, axis=1))) < 0.20
    if side_name == "front":
        # The white panel inside the physical card is retained; only connected
        # page background is removed.
        assert (
            int(
                np.sum(
                    np.all(np.asarray(side.processing_raster) >= 238, axis=2)
                )
            )
            > 5000
        )
    stages = side.source_diagnostics["stage_images"]
    assert {
        "raw_pdf_render",
        "detected_outer_background",
        "trim_bounding_box",
        "trimmed_pdf_raster",
        "final_card_detector_input",
    } <= set(stages)


def test_pdf_frame_trimming_is_not_applied_to_equivalent_png(tmp_path: Path) -> None:
    framed = Image.new("RGB", (1100, 750), (244, 241, 232))
    card = _rounded_card_asset("front")
    framed.paste(card, (100, 91), card)
    png = _write_image(tmp_path / "framed.png", framed)

    side = InputLoader().load_side(png, "front").card_side

    assert side.processing_raster.size == framed.size
    assert side.source_diagnostics["white_margin_trim_applied"] is False
    assert side.source_diagnostics["trim_box"] is None


def test_residual_top_strip_is_removed_on_bounded_second_pass(
    tmp_path: Path,
) -> None:
    pdf = _write_pdf(
        tmp_path / "residual-top-back.pdf",
        [_rounded_card_asset("back")],
        white_margin=True,
        inset_fraction=0.11,
    )
    side = InputLoader(debug=True).load_side(pdf, "back").card_side
    initial = side.source_diagnostics["initial_trim_box"]
    second = side.source_diagnostics["second_pass_trim_box"]

    assert initial is not None and second is not None
    assert second[1] > initial[1]
    assert side.source_diagnostics["residual_trim_offsets"][0] >= 3
    assert side.source_diagnostics["top_border_mean"] >= 245
    assert side.source_diagnostics["top_border_variance"] <= 5
    assert side.source_diagnostics["top_border_edge_density"] <= 0.01
    assert side.source_diagnostics["selected_top_edge_offset"] >= 3
    stages = side.source_diagnostics["stage_images"]
    assert {
        "first_trimmed_pdf_raster",
        "second_pass_residual_strip_detection",
    } <= set(stages)

    CardProcessingService(CardDetector(MALAYSIA_IC)).detect(side)
    correction = side.automatic_perspective_result
    assert correction is not None and correction.success
    assert correction.debug_info["post_rectify_top_white_ratio"] < 0.20
    refined = np.asarray(correction.refined_points)
    assert abs(float(refined[0, 1] - refined[1, 1])) < 2.0


def test_post_rectify_top_validation_moves_only_top_corners() -> None:
    pixels = np.full((650, 1000, 3), (255, 255, 255), dtype=np.uint8)
    pixels[20:] = (82, 142, 194)
    cv2.line(pixels, (0, 20), (999, 20), (35, 68, 102), 3)
    points = ((0.0, 0.0), (999.0, 0.0), (999.0, 649.0), (0.0, 649.0))

    result = PerspectiveCorrector(MALAYSIA_IC).correct(
        Image.fromarray(pixels),
        points,
        detector_confidence=0.86,
        method="automatic",
        refine=False,
        pdf_frame_background=(255, 255, 255),
    )

    assert result.success
    refined = np.asarray(result.refined_points)
    assert result.debug_info["selected_top_edge_offset"] > 0
    assert result.debug_info["post_rectify_top_white_ratio"] < 0.20
    assert refined[0, 1] > 5 and refined[1, 1] > 5
    assert refined[2] == pytest.approx(points[2], abs=0.01)
    assert refined[3] == pytest.approx(points[3], abs=0.01)
