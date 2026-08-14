from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from cardlayout.models.card_side import CardSide, SideName
from cardlayout.services.pdf_loader import PDFLoadError, PDFLoader

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SUPPORTED_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | {".pdf"}


class InputLoadError(ValueError):
    pass


@dataclass(slots=True)
class SideLoadResult:
    card_side: CardSide
    notice: str | None = None


@dataclass(slots=True)
class TwoPageLoadResult:
    front: CardSide
    back: CardSide | None
    page_count: int
    notice: str | None = None


class InputLoader:
    """Normalizes supported images and PDF pages into CardSide objects."""

    def __init__(self, pdf_loader: PDFLoader | None = None) -> None:
        self.pdf_loader = pdf_loader or PDFLoader()

    def load_side(self, path: str | Path, side: SideName) -> SideLoadResult:
        source = self._validate_path(path)
        if source.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            image = self._load_image(source)
            return SideLoadResult(self._make_side(side, source, "image", None, image))

        try:
            rendered = self.pdf_loader.render(source, [1])
        except PDFLoadError as exc:
            raise InputLoadError(str(exc)) from exc
        if not rendered.pages:
            raise InputLoadError("The PDF contains no usable pages")
        notice = None
        if rendered.page_count > 1:
            notice = f"This PDF has {rendered.page_count} pages; only Page 1 was used."
        page_number, image = rendered.pages[0]
        return SideLoadResult(
            self._make_side(side, source, "pdf", page_number, image), notice
        )

    def load_two_page_pdf(self, path: str | Path) -> TwoPageLoadResult:
        source = self._validate_path(path, pdf_only=True)
        try:
            rendered = self.pdf_loader.render(source, [1, 2])
        except PDFLoadError as exc:
            raise InputLoadError(str(exc)) from exc
        if not rendered.pages:
            raise InputLoadError("The PDF contains no usable pages")

        images = {number: image for number, image in rendered.pages}
        front = self._make_side("front", source, "pdf", 1, images[1])
        back = (
            self._make_side("back", source, "pdf", 2, images[2])
            if 2 in images
            else None
        )
        if rendered.page_count == 1:
            notice = "The PDF has one page. Page 1 was loaded as Front; Back is empty."
        elif rendered.page_count > 2:
            notice = (
                f"The PDF has {rendered.page_count} pages; only Page 1 and Page 2 "
                "were used."
            )
        else:
            notice = None
        return TwoPageLoadResult(front, back, rendered.page_count, notice)

    @staticmethod
    def _validate_path(path: str | Path, pdf_only: bool = False) -> Path:
        source = Path(path)
        if not source.is_file():
            raise InputLoadError("The selected file does not exist")
        suffix = source.suffix.lower()
        allowed = {".pdf"} if pdf_only else SUPPORTED_SUFFIXES
        if suffix not in allowed:
            expected = "a PDF" if pdf_only else "a JPG, JPEG, PNG, or PDF"
            raise InputLoadError(f"Please choose {expected} file")
        return source

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        try:
            with Image.open(path) as source:
                source.load()
                return ImageOps.exif_transpose(source).convert("RGB").copy()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InputLoadError("The image could not be opened") from exc

    @staticmethod
    def _make_side(
        side: SideName,
        path: Path,
        source_type: str,
        source_page: int | None,
        image: Image.Image,
    ) -> CardSide:
        original = image.copy()
        return CardSide(
            side=side,
            source_path=path,
            source_type=source_type,  # type: ignore[arg-type]
            source_page=source_page,
            original_image=original,
            processed_image=original.copy(),
        )

