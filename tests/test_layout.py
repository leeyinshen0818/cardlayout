import pytest

from cardlayout.models.card_size import MALAYSIA_IC
from cardlayout.models.layout import A4_LAYOUT
from cardlayout.services.layout_engine import LayoutEngine


@pytest.fixture
def engine() -> LayoutEngine:
    return LayoutEngine(A4_LAYOUT, MALAYSIA_IC)


def test_a4_and_malaysia_ic_dimensions(engine: LayoutEngine) -> None:
    page = engine.calculate()
    assert (page.page_width_mm, page.page_height_mm) == (210.0, 297.0)
    assert (page.front.width, page.front.height) == (85.6, 54.0)
    assert (page.back.width, page.back.height) == (85.6, 54.0)


def test_cards_are_horizontally_centered(engine: LayoutEngine) -> None:
    page = engine.calculate()
    expected_x = (210.0 - 85.6) / 2
    assert page.front.x == pytest.approx(expected_x)
    assert page.back.x == pytest.approx(expected_x)
    assert page.front.x + page.front.width / 2 == pytest.approx(105.0)


def test_vertical_positions_and_gap(engine: LayoutEngine) -> None:
    page = engine.calculate()
    assert page.front.y == pytest.approx(51.6)
    assert page.front.bottom == pytest.approx(105.6)
    assert page.back.y == pytest.approx(125.6)
    assert page.back.bottom == pytest.approx(179.6)
    assert page.back.y - page.front.bottom == pytest.approx(20.0)


def test_front_and_back_can_move_independently(engine: LayoutEngine) -> None:
    engine.adjust_vertical_offset("front", 7.0)
    engine.adjust_vertical_offset("back", -4.0)
    page = engine.calculate()
    assert page.front.y == pytest.approx(58.6)
    assert page.back.y == pytest.approx(121.6)
    assert engine.vertical_offset("front") == 7.0
    assert engine.vertical_offset("back") == -4.0


def test_vertical_adjustment_stays_inside_a4(engine: LayoutEngine) -> None:
    engine.adjust_vertical_offset("front", -1000.0)
    engine.adjust_vertical_offset("back", 1000.0)
    page = engine.calculate()
    assert page.front.y == pytest.approx(0.0)
    assert page.back.bottom == pytest.approx(297.0)


def test_vertical_position_can_be_reset(engine: LayoutEngine) -> None:
    engine.adjust_vertical_offset("front", 12.0)
    engine.reset_vertical_offset("front")
    assert engine.vertical_offset("front") == 0.0
    assert engine.calculate().front.y == pytest.approx(51.6)


def test_a4_pixel_dimensions_at_300_dpi(engine: LayoutEngine) -> None:
    assert engine.page_pixels(300) == (2480, 3508)


def test_pdf_and_raster_rects_share_normalized_geometry(engine: LayoutEngine) -> None:
    page = engine.calculate()
    pixel_rect = engine.rect_at_dpi(page.front, 300)
    point_rect = engine.rect_in_points(page.front)
    page_pixels = engine.page_pixels(300)
    page_points = engine.page_points()
    pixel_normalized = tuple(
        value / page_pixels[index % 2] for index, value in enumerate(pixel_rect)
    )
    point_normalized = tuple(
        value / page_points[index % 2] for index, value in enumerate(point_rect)
    )
    assert pixel_normalized == pytest.approx(point_normalized, abs=0.0003)
