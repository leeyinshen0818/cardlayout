from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from cardlayout.services.raster_normalizer import normalize_raster_for_detection


class PDFLoadError(ValueError):
    pass


@dataclass(slots=True)
class RenderedPDF:
    page_count: int
    pages: list[tuple[int, Image.Image]]
    page_diagnostics: dict[int, dict[str, object]]


class PDFLoader:
    """Renders selected PDF pages directly to memory using PyMuPDF."""

    def __init__(
        self,
        dpi: float = 240.0,
        maximum_dpi: float = 300.0,
        minimum_short_edge_px: int = 1400,
        maximum_long_edge_px: int = 4200,
    ) -> None:
        self.dpi = dpi
        self.maximum_dpi = maximum_dpi
        self.minimum_short_edge_px = minimum_short_edge_px
        self.maximum_long_edge_px = maximum_long_edge_px

    def render(self, path: str | Path, page_numbers: list[int]) -> RenderedPDF:
        try:
            import pymupdf as fitz
        except ImportError as exc:  # pragma: no cover - installation problem
            raise PDFLoadError("PyMuPDF is required to open PDF files") from exc

        source = Path(path)
        try:
            document = fitz.open(source)
            try:
                page_count = document.page_count
                if page_count == 0:
                    raise PDFLoadError("The PDF contains no pages")
                pages: list[tuple[int, Image.Image]] = []
                diagnostics: dict[int, dict[str, object]] = {}
                for page_number in page_numbers:
                    if not 1 <= page_number <= page_count:
                        continue
                    page = document.load_page(page_number - 1)
                    page_width = float(page.rect.width)
                    page_height = float(page.rect.height)
                    short_points = max(1.0, min(page_width, page_height))
                    long_points = max(page_width, page_height)
                    effective_dpi = max(
                        self.dpi,
                        self.minimum_short_edge_px * 72.0 / short_points,
                    )
                    effective_dpi = min(effective_dpi, self.maximum_dpi)
                    if long_points * effective_dpi / 72.0 > self.maximum_long_edge_px:
                        effective_dpi = min(
                            effective_dpi,
                            self.maximum_long_edge_px * 72.0 / long_points,
                        )
                    initial_dpi = effective_dpi

                    def render_at(render_dpi: float):
                        matrix = fitz.Matrix(render_dpi / 72.0, render_dpi / 72.0)
                        rendered_pixmap = page.get_pixmap(
                            matrix=matrix,
                            colorspace=fitz.csRGB,
                            alpha=False,
                        )
                        rendered_image = Image.frombytes(
                            "RGB",
                            (rendered_pixmap.width, rendered_pixmap.height),
                            rendered_pixmap.samples,
                        ).copy()
                        return rendered_pixmap, rendered_image

                    pixmap, image = render_at(effective_dpi)
                    probe_image = image.copy()
                    probe_image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
                    content_probe = normalize_raster_for_detection(
                        probe_image, trim_pdf_whitespace=True
                    )
                    maximum_page_dpi = min(
                        self.maximum_dpi,
                        self.maximum_long_edge_px * 72.0 / long_points,
                    )
                    adaptive_rerender = bool(
                        content_probe.trim_applied
                        and min(content_probe.image.size) < 350
                        and maximum_page_dpi > effective_dpi + 1.0
                    )
                    if adaptive_rerender:
                        effective_dpi = maximum_page_dpi
                        pixmap, image = render_at(effective_dpi)
                    zoom = effective_dpi / 72.0
                    pages.append((page_number, image))
                    diagnostics[page_number] = {
                        "pdf_page_width_points": page_width,
                        "pdf_page_height_points": page_height,
                        "pdf_mediabox": tuple(float(value) for value in page.mediabox),
                        "pdf_cropbox": tuple(float(value) for value in page.cropbox),
                        "pdf_rotation": int(page.rotation) % 360,
                        "render_zoom": round(zoom, 5),
                        "effective_render_dpi": round(effective_dpi, 2),
                        "initial_render_dpi": round(initial_dpi, 2),
                        "adaptive_rerender_applied": adaptive_rerender,
                        "render_width": pixmap.width,
                        "render_height": pixmap.height,
                        "render_color_mode": "RGB",
                        "render_alpha": False,
                    }
                return RenderedPDF(
                    page_count=page_count,
                    pages=pages,
                    page_diagnostics=diagnostics,
                )
            finally:
                document.close()
        except PDFLoadError:
            raise
        except Exception as exc:
            raise PDFLoadError("The PDF could not be opened or rendered") from exc
