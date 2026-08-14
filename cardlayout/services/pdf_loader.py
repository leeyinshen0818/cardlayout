from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


class PDFLoadError(ValueError):
    pass


@dataclass(slots=True)
class RenderedPDF:
    page_count: int
    pages: list[tuple[int, Image.Image]]


class PDFLoader:
    """Renders selected PDF pages directly to memory using PyMuPDF."""

    def __init__(self, dpi: float = 200.0) -> None:
        self.dpi = dpi

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
                matrix = fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)
                pages: list[tuple[int, Image.Image]] = []
                for page_number in page_numbers:
                    if not 1 <= page_number <= page_count:
                        continue
                    page = document.load_page(page_number - 1)
                    pixmap = page.get_pixmap(
                        matrix=matrix,
                        colorspace=fitz.csRGB,
                        alpha=False,
                    )
                    image = Image.frombytes(
                        "RGB", (pixmap.width, pixmap.height), pixmap.samples
                    ).copy()
                    pages.append((page_number, image))
                return RenderedPDF(page_count=page_count, pages=pages)
            finally:
                document.close()
        except PDFLoadError:
            raise
        except Exception as exc:
            raise PDFLoadError("The PDF could not be opened or rendered") from exc
