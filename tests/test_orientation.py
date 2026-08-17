from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from cardlayout.models.card_side import CardSide
from cardlayout.models.image_correction import ImageCorrectionState
from cardlayout.models.orientation import OrientationState
from cardlayout.models.perspective import PerspectiveResult
from cardlayout.services.image_corrections import apply_image_correction
from cardlayout.services.orientation import OrientationAnalyzer, apply_orientation


def _pattern() -> Image.Image:
    image = Image.new("RGB", (4, 3))
    image.putdata(
        [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (40, 50, 60), (70, 80, 90),
            (100, 110, 120), (130, 140, 150), (160, 170, 180), (190, 200, 210),
        ]
    )
    return image


def _side(name: str = "front", source_type: str = "image") -> CardSide:
    image = _pattern()
    return CardSide(
        side=name,  # type: ignore[arg-type]
        source_path=Path(f"{name}.png"),
        source_type=source_type,  # type: ignore[arg-type]
        source_page=1 if source_type == "pdf" else None,
        original_image=image,
        processed_image=image.copy(),
    )


@pytest.mark.parametrize(
    ("state", "transpose"),
    [
        (OrientationState(flip_horizontal=True), Image.Transpose.FLIP_LEFT_RIGHT),
        (OrientationState(flip_vertical=True), Image.Transpose.FLIP_TOP_BOTTOM),
        (
            OrientationState(flip_horizontal=True, flip_vertical=True),
            Image.Transpose.ROTATE_180,
        ),
    ],
)
def test_orientation_operations_match_exact_non_destructive_transforms(
    state: OrientationState, transpose: Image.Transpose
) -> None:
    source = _pattern()
    before = source.tobytes()

    result = apply_orientation(source, state)

    assert result.tobytes() == source.transpose(transpose).tobytes()
    assert source.tobytes() == before


def test_orientation_actions_toggle_and_reset_to_normal() -> None:
    state = OrientationState().apply_action("flip_horizontal")
    assert state.label == "Flipped Horizontal"
    state = state.apply_action("flip_vertical")
    assert state.label == "Rotated 180°"
    state = state.apply_action("rotate_180")
    assert state.is_normal
    assert state.apply_action("reset").is_normal


def test_orientation_precedes_appearance_corrections_and_never_accumulates() -> None:
    side = _side()
    geometry = _pattern().resize((40, 30), Image.Resampling.NEAREST)
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
    state = OrientationState(flip_horizontal=True)
    correction = ImageCorrectionState(sharpen="sharp", tone="bright_10")
    original_bytes = side.original_image.tobytes()

    side.apply_orientation(state)
    side.apply_image_correction(correction)
    first = side.best_image.tobytes()
    side.apply_orientation(OrientationState(flip_vertical=True))
    side.apply_orientation(state)

    expected = apply_image_correction(apply_orientation(geometry, state), correction)
    assert side.best_image.tobytes() == first == expected.tobytes()
    assert side.original_image.tobytes() == original_bytes
    assert side.geometry_image is geometry


def test_front_back_and_source_types_share_independent_orientation_state() -> None:
    front = _side("front", "image")
    back = _side("back", "pdf")

    front.apply_orientation(OrientationState(flip_horizontal=True))
    back.apply_orientation(OrientationState(flip_vertical=True))

    assert front.orientation_state.label == "Flipped Horizontal"
    assert back.orientation_state.label == "Flipped Vertical"
    assert front.best_image.tobytes() != front.original_image.tobytes()
    assert back.best_image.tobytes() != back.original_image.tobytes()


def test_side_reset_restores_automatic_geometry_appearance_and_orientation() -> None:
    side = _side()
    automatic = _pattern().resize((40, 30), Image.Resampling.NEAREST)
    manual = Image.new("RGB", automatic.size, (30, 60, 90))
    side.apply_automatic_correction(
        PerspectiveResult(success=True, rectified_image=automatic, method="automatic")
    )
    side.apply_manual_correction(
        PerspectiveResult(success=True, rectified_image=manual, method="manual")
    )
    side.apply_orientation(OrientationState(flip_horizontal=True))
    side.apply_image_correction(ImageCorrectionState(tone="bright_20"))

    side.reset_user_edits()

    assert not side.has_manual_correction
    assert side.orientation_state.is_normal
    assert side.image_correction_state.is_normal
    assert side.oriented_image is None
    assert side.corrected_image is None
    assert side.best_image is automatic


def test_automatic_orientation_abstains_without_reliable_local_text_recognition() -> None:
    analysis = OrientationAnalyzer().analyze(_pattern())

    assert not analysis.applied
    assert analysis.confidence == 0.0
    assert analysis.recommended_state.is_normal
    assert "unchanged" in analysis.reason
