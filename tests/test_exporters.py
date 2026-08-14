from pathlib import Path

import pytest
from PIL import Image

from cardlayout.models.card_size import MALAYSIA_IC
from cardlayout.models.layout import A4_LAYOUT
from cardlayout.models.perspective import PerspectiveResult
from cardlayout.services.image_exporter import ImageExporter
from cardlayout.services.input_loader import InputLoader
from cardlayout.services.layout_engine import LayoutEngine
from cardlayout.services.page_renderer import PageRenderer
from cardlayout.services.pdf_exporter import PDFExporter


@pytest.fixture
def export_setup(sample_jpg: Path, sample_png: Path):
    engine = LayoutEngine(A4_LAYOUT, MALAYSIA_IC)
    renderer = PageRenderer(engine)
    loader = InputLoader()
    front = loader.load_side(sample_jpg, "front").card_side
    back = loader.load_side(sample_png, "back").card_side
    return engine, renderer, front, back


def test_jpg_is_300_dpi_a4(tmp_path: Path, export_setup) -> None:
    _, renderer, front, back = export_setup
    output = tmp_path / "layout.jpg"
    ImageExporter(renderer).export(output, front, back)
    assert output.is_file() and output.stat().st_size > 0
    with Image.open(output) as page:
        assert page.size == (2480, 3508)
        assert page.info["dpi"] == pytest.approx((300, 300), abs=1)


def test_pdf_is_real_a4_page(tmp_path: Path, export_setup) -> None:
    import pymupdf as fitz

    _, renderer, front, back = export_setup
    output = tmp_path / "layout.pdf"
    PDFExporter(renderer).export(output, front, back)
    assert output.is_file() and output.stat().st_size > 0
    with fitz.open(output) as document:
        assert document.page_count == 1
        rect = document[0].rect
        assert rect.width == pytest.approx(595.28, abs=0.05)
        assert rect.height == pytest.approx(841.89, abs=0.05)


def test_pdf_and_jpg_put_both_sides_at_same_positions(tmp_path: Path, export_setup) -> None:
    import pymupdf as fitz

    engine, renderer, front, back = export_setup
    engine.adjust_vertical_offset("front", 8.0)
    engine.adjust_vertical_offset("back", -6.0)
    jpg_path = tmp_path / "layout.jpg"
    pdf_path = tmp_path / "layout.pdf"
    ImageExporter(renderer).export(jpg_path, front, back)
    PDFExporter(renderer).export(pdf_path, front, back)

    with Image.open(jpg_path) as jpg:
        jpg_pixels = jpg.convert("RGB")
        layout = engine.calculate()
        centers = []
        for rect in (layout.front, layout.back):
            bounds = engine.rect_at_dpi(rect, 300)
            centers.append(((bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2))
        jpg_colors = [jpg_pixels.getpixel(center) for center in centers]

    with fitz.open(pdf_path) as document:
        pix = document[0].get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False)
        pdf_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pdf_colors = [pdf_image.getpixel(center) for center in centers]

    assert jpg_colors[0][0] > 150 and jpg_colors[0][1] < 70
    assert pdf_colors[0][0] > 150 and pdf_colors[0][1] < 70
    assert jpg_colors[1][2] > jpg_colors[1][0]
    assert pdf_colors[1][2] > pdf_colors[1][0]


def test_pdf_and_jpg_use_manual_corrected_image_as_best_stage(
    tmp_path: Path, export_setup
) -> None:
    import pymupdf as fitz

    engine, renderer, front, _ = export_setup
    automatic_image = Image.new("RGB", (856, 540), (20, 190, 60))
    manual_image = Image.new("RGB", (856, 540), (245, 190, 25))
    front.apply_automatic_correction(
        PerspectiveResult(
            success=True,
            rectified_image=automatic_image,
            confidence=0.9,
            confidence_level="high",
            status="corrected",
            method="automatic",
        )
    )
    front.apply_manual_correction(
        PerspectiveResult(
            success=True,
            rectified_image=manual_image,
            confidence=0.95,
            confidence_level="high",
            status="corrected",
            method="manual",
        )
    )
    jpg_path = tmp_path / "corrected-layout.jpg"
    pdf_path = tmp_path / "corrected-layout.pdf"
    ImageExporter(renderer).export(jpg_path, front, None)
    PDFExporter(renderer).export(pdf_path, front, None)
    bounds = engine.rect_at_dpi(engine.calculate().front, 300)
    center = ((bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2)

    with Image.open(jpg_path) as jpg:
        jpg_color = jpg.convert("RGB").getpixel(center)
    with fitz.open(pdf_path) as document:
        pix = document[0].get_pixmap(
            matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False
        )
        pdf_color = Image.frombytes(
            "RGB", (pix.width, pix.height), pix.samples
        ).getpixel(center)

    assert jpg_color[0] > 200 and jpg_color[1] > 140 and jpg_color[2] < 70
    assert pdf_color[0] > 200 and pdf_color[1] > 140 and pdf_color[2] < 70
