from pathlib import Path

import pytest

from cardlayout.services.input_loader import InputLoadError, InputLoader


def test_jpg_loads_and_normalizes(sample_jpg: Path) -> None:
    result = InputLoader().load_side(sample_jpg, "front")
    side = result.card_side
    assert side.side == "front"
    assert side.source_type == "image"
    assert side.source_page is None
    assert side.processed_image.mode == "RGB"
    assert side.processed_image.size == (428, 270)


def test_png_loads_and_flattens_transparency(sample_png: Path) -> None:
    side = InputLoader().load_side(sample_png, "back").card_side
    assert side.side == "back"
    assert side.source_type == "image"
    assert side.processed_image.mode == "RGB"
    # The source is RGBA (20, 90, 190, 210); transparency is composited onto
    # white rather than silently discarded by a plain RGB conversion.
    assert side.processed_image.getpixel((0, 0)) == pytest.approx(
        (61, 119, 201), abs=1
    )


def test_one_page_pdf_loads_front_and_leaves_back_empty(one_page_pdf: Path) -> None:
    result = InputLoader().load_two_page_pdf(one_page_pdf)
    assert result.front.source_page == 1
    assert result.back is None
    assert result.page_count == 1
    assert result.notice and "Back is empty" in result.notice


def test_two_page_pdf_maps_pages_to_front_and_back(two_page_pdf: Path) -> None:
    result = InputLoader().load_two_page_pdf(two_page_pdf)
    assert result.front.side == "front"
    assert result.front.source_page == 1
    assert result.back is not None
    assert result.back.side == "back"
    assert result.back.source_page == 2
    assert result.notice is None


def test_more_than_two_pdf_pages_reports_first_two_only(three_page_pdf: Path) -> None:
    result = InputLoader().load_two_page_pdf(three_page_pdf)
    assert result.page_count == 3
    assert result.front.source_page == 1
    assert result.back and result.back.source_page == 2
    assert result.notice and "only Page 1 and Page 2" in result.notice


def test_separate_pdf_uses_first_page(three_page_pdf: Path) -> None:
    result = InputLoader().load_side(three_page_pdf, "back")
    assert result.card_side.source_page == 1
    assert result.card_side.side == "back"
    assert result.notice and "only Page 1" in result.notice


def test_invalid_file_is_rejected(tmp_path: Path) -> None:
    invalid = tmp_path / "not-an-image.jpg"
    invalid.write_text("invalid", encoding="utf-8")
    with pytest.raises(InputLoadError, match="could not be opened"):
        InputLoader().load_side(invalid, "front")


def test_unsupported_format_is_rejected(tmp_path: Path) -> None:
    unsupported = tmp_path / "card.bmp"
    unsupported.write_bytes(b"BM")
    with pytest.raises(InputLoadError, match="JPG"):
        InputLoader().load_side(unsupported, "front")
