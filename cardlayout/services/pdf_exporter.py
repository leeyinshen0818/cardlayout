from __future__ import annotations

from io import BytesIO
from pathlib import Path

from cardlayout.models.card_side import CardSide
from cardlayout.models.layout import MmRect
from cardlayout.services.image_exporter import ExportError
from cardlayout.services.page_renderer import PageRenderer


class PDFExporter:
    def __init__(self, renderer: PageRenderer, artwork_dpi: int = 300) -> None:
        self.renderer = renderer
        self.engine = renderer.engine
        self.artwork_dpi = artwork_dpi

    def export(
        self,
        path: str | Path,
        front: CardSide | None,
        back: CardSide | None,
    ) -> None:
        try:
            import pymupdf as fitz
        except ImportError as exc:  # pragma: no cover - installation problem
            raise ExportError("PyMuPDF is required to export PDF files") from exc

        destination = Path(path)
        document = fitz.open()
        try:
            page_width, page_height = self.engine.page_points()
            page = document.new_page(width=page_width, height=page_height)
            layout = self.engine.calculate()
            if front is not None:
                self._insert_card(page, layout.front, front, "FRONT")
            if back is not None:
                self._insert_card(page, layout.back, back, "BACK")
            pdf_data = document.tobytes(garbage=4, deflate=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(pdf_data)
        except (OSError, ValueError, RuntimeError) as exc:
            raise ExportError("The PDF could not be saved") from exc
        finally:
            document.close()

    def _insert_card(
        self,
        page: object,
        rect: MmRect,
        side: CardSide | None,
        label: str,
    ) -> None:
        import pymupdf as fitz

        pixel_rect = self.engine.rect_at_dpi(rect, self.artwork_dpi)
        size = (pixel_rect[2] - pixel_rect[0], pixel_rect[3] - pixel_rect[1])
        card = self.renderer.render_card(
            side,
            size,
            label,
            border_width=max(1, int(round(self.artwork_dpi / 150))),
        )
        stream = BytesIO()
        card.save(stream, format="PNG", optimize=True)
        page.insert_image(  # type: ignore[attr-defined]
            fitz.Rect(*self.engine.rect_in_points(rect)),
            stream=stream.getvalue(),
            keep_proportion=False,
        )
