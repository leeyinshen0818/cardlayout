from pathlib import Path

import pytest
from PIL import Image

from cardlayout.models.card_size import MALAYSIA_IC
from cardlayout.models.layout import A4_LAYOUT
from cardlayout.models.image_correction import ImageCorrectionState
from cardlayout.models.orientation import OrientationState
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


def test_front_only_preview_renderer_and_exports_leave_back_area_blank(
    tmp_path: Path, export_setup
) -> None:
    import pymupdf as fitz

    engine, renderer, front, _ = export_setup
    layout = engine.calculate()
    back_bounds = engine.rect_at_dpi(layout.back, 300)
    jpg_path = tmp_path / "front-only.jpg"
    pdf_path = tmp_path / "front-only.pdf"

    rendered = renderer.render_page(front, None, 300)
    ImageExporter(renderer).export(jpg_path, front, None)
    PDFExporter(renderer).export(pdf_path, front, None)

    def back_region(image: Image.Image) -> Image.Image:
        return image.convert("RGB").crop(back_bounds)

    assert set(back_region(rendered).getdata()) == {(255, 255, 255)}
    with Image.open(jpg_path) as jpg:
        assert min(min(pixel) for pixel in back_region(jpg).getdata()) >= 250
    with fitz.open(pdf_path) as document:
        pixmap = document[0].get_pixmap(
            matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False
        )
        page = Image.frombytes(
            "RGB", (pixmap.width, pixmap.height), pixmap.samples
        )
        assert min(min(pixel) for pixel in back_region(page).getdata()) >= 250


def test_independent_orientation_is_used_by_jpg_and_pdf_exports(
    tmp_path: Path, export_setup
) -> None:
    import pymupdf as fitz

    engine, renderer, front, back = export_setup
    front_geometry = Image.new("RGB", (856, 540), (230, 30, 20))
    front_geometry.paste((20, 210, 40), (428, 0, 856, 540))
    back_geometry = Image.new("RGB", (856, 540), (20, 40, 230))
    back_geometry.paste((235, 220, 20), (0, 270, 856, 540))
    front.apply_automatic_correction(
        PerspectiveResult(success=True, rectified_image=front_geometry)
    )
    back.apply_automatic_correction(
        PerspectiveResult(success=True, rectified_image=back_geometry)
    )
    front.apply_orientation(OrientationState(flip_horizontal=True))
    back.apply_orientation(OrientationState(flip_vertical=True))

    jpg_path = tmp_path / "oriented.jpg"
    pdf_path = tmp_path / "oriented.pdf"
    ImageExporter(renderer).export(jpg_path, front, back)
    PDFExporter(renderer).export(pdf_path, front, back)

    layout = engine.calculate()
    front_bounds = engine.rect_at_dpi(layout.front, 300)
    back_bounds = engine.rect_at_dpi(layout.back, 300)
    samples = (
        (
            front_bounds[0] + (front_bounds[2] - front_bounds[0]) // 4,
            (front_bounds[1] + front_bounds[3]) // 2,
        ),
        (
            (back_bounds[0] + back_bounds[2]) // 2,
            back_bounds[1] + (back_bounds[3] - back_bounds[1]) // 4,
        ),
    )

    with Image.open(jpg_path) as jpg:
        jpg_colors = [jpg.convert("RGB").getpixel(point) for point in samples]
    with fitz.open(pdf_path) as document:
        pixmap = document[0].get_pixmap(
            matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False
        )
        pdf_page = Image.frombytes(
            "RGB", (pixmap.width, pixmap.height), pixmap.samples
        )
        pdf_colors = [pdf_page.getpixel(point) for point in samples]

    for color in (jpg_colors[0], pdf_colors[0]):
        assert color[1] > 170 and color[0] < 70
    for color in (jpg_colors[1], pdf_colors[1]):
        assert color[0] > 190 and color[1] > 180 and color[2] < 70


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


def test_normal_correction_preserves_existing_render_exactly(export_setup) -> None:
    _, renderer, front, back = export_setup
    before = renderer.render_page(front, back, 150)

    front.apply_image_correction(ImageCorrectionState())
    back.apply_image_correction(ImageCorrectionState())
    after = renderer.render_page(front, back, 150)

    assert after.size == before.size
    assert after.tobytes() == before.tobytes()


def test_independent_front_and_back_presets_are_used_by_jpg_and_pdf(
    tmp_path: Path, export_setup
) -> None:
    import pymupdf as fitz

    engine, renderer, front, back = export_setup
    baseline = renderer.render_page(front, back, 300)
    layout = engine.calculate()
    centers = []
    for rect in (layout.front, layout.back):
        bounds = engine.rect_at_dpi(rect, 300)
        centers.append(((bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2))
    baseline_colors = [baseline.getpixel(center) for center in centers]

    front.apply_image_correction(ImageCorrectionState(tone="bright_10"))
    back.apply_image_correction(
        ImageCorrectionState(sharpen="sharp", tone="bright_20")
    )
    jpg_path = tmp_path / "corrected-presets.jpg"
    pdf_path = tmp_path / "corrected-presets.pdf"
    ImageExporter(renderer).export(jpg_path, front, back)
    PDFExporter(renderer).export(pdf_path, front, back)

    with Image.open(jpg_path) as jpg:
        assert jpg.size == baseline.size
        jpg_colors = [jpg.convert("RGB").getpixel(center) for center in centers]
    with fitz.open(pdf_path) as document:
        pix = document[0].get_pixmap(
            matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False
        )
        pdf_page = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pdf_colors = [pdf_page.getpixel(center) for center in centers]

    for index in range(2):
        assert sum(jpg_colors[index]) > sum(baseline_colors[index]) + 8
        assert sum(pdf_colors[index]) > sum(baseline_colors[index]) + 8
    assert front.image_correction_state.tone == "bright_10"
    assert back.image_correction_state.tone == "bright_20"
    assert front.image_correction_state.sharpen == "normal"
    assert back.image_correction_state.sharpen == "sharp"
