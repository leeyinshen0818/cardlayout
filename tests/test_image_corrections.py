from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cardlayout.models.card_side import CardSide
from cardlayout.models.image_correction import ImageCorrectionState
from cardlayout.models.perspective import PerspectiveResult
from cardlayout.services.image_corrections import apply_image_correction


def _detail_image() -> Image.Image:
    pixels = np.full((180, 280, 3), (105, 125, 145), dtype=np.uint8)
    for x in range(0, 280, 14):
        cv2.line(pixels, (x, 0), (x, 179), (35, 60, 90), 2)
    cv2.rectangle(pixels, (38, 35), (238, 145), (185, 205, 220), -1)
    cv2.putText(
        pixels,
        "CARD 123",
        (55, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (25, 55, 90),
        3,
        cv2.LINE_AA,
    )
    return Image.fromarray(pixels)


def _edge_energy(image: Image.Image) -> float:
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _side(side: str, image: Image.Image) -> CardSide:
    return CardSide(
        side=side,  # type: ignore[arg-type]
        source_path=Path(f"{side}.png"),
        source_type="image",
        source_page=None,
        original_image=image,
        processed_image=image.copy(),
    )


def test_normal_correction_is_pixel_identical_and_non_destructive() -> None:
    source = _detail_image()
    before = source.tobytes()

    result = apply_image_correction(source, ImageCorrectionState())

    assert result is not source
    assert result.tobytes() == before
    assert source.tobytes() == before


def test_soft_sharp_and_sharper_presets_have_predictable_strength() -> None:
    source = _detail_image()
    soft = apply_image_correction(source, ImageCorrectionState(sharpen="soft"))
    sharp = apply_image_correction(source, ImageCorrectionState(sharpen="sharp"))
    sharper = apply_image_correction(source, ImageCorrectionState(sharpen="sharper"))

    assert soft.tobytes() != source.tobytes()
    assert sharp.tobytes() != source.tobytes()
    assert _edge_energy(soft) < _edge_energy(source)
    assert _edge_energy(sharp) > _edge_energy(source)
    assert _edge_energy(sharper) > _edge_energy(sharp)


def test_brightness_and_contrast_presets_are_ordered() -> None:
    source = _detail_image()
    normal = np.asarray(source).astype(np.float32)
    bright_10 = np.asarray(
        apply_image_correction(source, ImageCorrectionState(tone="bright_10"))
    ).astype(np.float32)
    bright_20 = np.asarray(
        apply_image_correction(source, ImageCorrectionState(tone="bright_20"))
    ).astype(np.float32)
    contrast = np.asarray(
        apply_image_correction(source, ImageCorrectionState(tone="bright_contrast"))
    ).astype(np.float32)

    assert bright_10.mean() > normal.mean()
    assert bright_20.mean() > bright_10.mean()
    assert contrast.std() > bright_10.std()


def test_front_and_back_keep_independent_correction_state() -> None:
    source = _detail_image()
    front = _side("front", source)
    back = _side("back", source)

    front.apply_image_correction(
        ImageCorrectionState(sharpen="sharp", tone="bright_10")
    )
    back.apply_image_correction(ImageCorrectionState(tone="bright_contrast"))

    assert front.image_correction_state.sharpen == "sharp"
    assert front.image_correction_state.tone == "bright_10"
    assert back.image_correction_state.sharpen == "normal"
    assert back.image_correction_state.tone == "bright_contrast"
    assert front.best_image.tobytes() != back.best_image.tobytes()


def test_corrections_start_from_geometry_source_and_never_accumulate() -> None:
    original = Image.new("RGB", (280, 180), (20, 30, 40))
    geometry = _detail_image()
    side = _side("front", original)
    side.apply_automatic_correction(
        PerspectiveResult(
            success=True,
            rectified_image=geometry,
            confidence=0.9,
            confidence_level="high",
            status="corrected",
            method="automatic",
        )
    )
    original_bytes = side.original_image.tobytes()
    sharp_state = ImageCorrectionState(sharpen="sharp", tone="bright_10")

    side.apply_image_correction(sharp_state)
    first = side.best_image.tobytes()
    side.apply_image_correction(ImageCorrectionState(sharpen="soft", tone="bright_20"))
    side.apply_image_correction(sharp_state)

    expected = apply_image_correction(geometry, sharp_state)
    assert side.best_image.tobytes() == first == expected.tobytes()
    assert side.original_image.tobytes() == original_bytes
    assert side.geometry_image is geometry


def test_reset_restores_automatic_geometry_and_normal_corrections() -> None:
    source = _detail_image()
    automatic_image = source.copy()
    manual_image = Image.new("RGB", source.size, (220, 180, 80))
    side = _side("front", source)
    side.apply_automatic_correction(
        PerspectiveResult(
            success=True,
            rectified_image=automatic_image,
            confidence=0.9,
            confidence_level="high",
            status="corrected",
            method="automatic",
        )
    )
    side.apply_manual_correction(
        PerspectiveResult(
            success=True,
            rectified_image=manual_image,
            confidence=0.95,
            confidence_level="high",
            status="corrected",
            method="manual",
        )
    )
    side.apply_image_correction(
        ImageCorrectionState(sharpen="sharper", tone="bright_20")
    )

    side.reset_user_edits()

    assert not side.has_manual_correction
    assert side.image_correction_state.is_normal
    assert side.corrected_image is None
    assert side.best_image is automatic_image
