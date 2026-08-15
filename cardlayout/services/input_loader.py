from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from cardlayout.models.card_side import CardSide, SideName
from cardlayout.services.pdf_loader import PDFLoadError, PDFLoader
from cardlayout.services.raster_normalizer import normalize_raster_for_detection

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

    def __init__(self, pdf_loader: PDFLoader | None = None, debug: bool = False) -> None:
        self.pdf_loader = pdf_loader or PDFLoader()
        self.debug = debug

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
            self._make_side(
                side,
                source,
                "pdf",
                page_number,
                image,
                rendered.page_diagnostics.get(page_number),
            ),
            notice,
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
        front = self._make_side(
            "front", source, "pdf", 1, images[1], rendered.page_diagnostics.get(1)
        )
        back = (
            self._make_side(
                "back", source, "pdf", 2, images[2], rendered.page_diagnostics.get(2)
            )
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
                return source.copy()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InputLoadError("The image could not be opened") from exc

    def _make_side(
        self,
        side: SideName,
        path: Path,
        source_type: str,
        source_page: int | None,
        image: Image.Image,
        pdf_diagnostics: dict[str, object] | None = None,
    ) -> CardSide:
        raw = image.copy()
        normalized = normalize_raster_for_detection(
            raw, trim_pdf_whitespace=source_type == "pdf"
        )
        detector_input = normalized.image.copy()
        diagnostics: dict[str, object] = {
            "source_type": source_type,
            "source_path": str(path),
            "source_page": source_page,
            "original_raster_width": raw.width,
            "original_raster_height": raw.height,
            "normalized_width": detector_input.width,
            "normalized_height": detector_input.height,
            "white_margin_trim_applied": normalized.trim_applied,
            "trim_box": normalized.trim_box,
            "initial_trim_box": normalized.initial_trim_box,
            "second_pass_trim_box": normalized.second_pass_trim_box,
            "residual_trim_offsets": normalized.residual_trim_offsets,
            "outer_background_color": normalized.background_color,
            "outer_background_tolerance": normalized.background_tolerance,
            "outer_background_confidence": round(
                normalized.background_confidence, 4
            ),
            "trim_safety_margin_fraction": 0.025,
            "trim_safety_margin_fraction_used": (
                normalized.safety_margin_fraction_used
            ),
            "detector_input_width": detector_input.width,
            "detector_input_height": detector_input.height,
            "detector_input_mode": detector_input.mode,
        }
        if pdf_diagnostics:
            diagnostics.update(pdf_diagnostics)
        if normalized.residual_metrics:
            diagnostics.update(normalized.residual_metrics)
        if self.debug:
            stages: dict[str, Image.Image] = {
                "detector_input": detector_input.copy(),
            }
            if source_type == "pdf":
                stages["raw_pdf_render"] = raw.copy()
                stages["detected_outer_background"] = (
                    normalized.outer_background_mask.copy()
                    if normalized.outer_background_mask is not None
                    else Image.new("L", raw.size, 0)
                )
                stages["trim_bounding_box"] = (
                    normalized.trim_overlay.copy()
                    if normalized.trim_overlay is not None
                    else raw.copy()
                )
                stages["trimmed_pdf_raster"] = detector_input.copy()
                if normalized.first_trimmed_image is not None:
                    stages["first_trimmed_pdf_raster"] = (
                        normalized.first_trimmed_image.copy()
                    )
                if normalized.residual_trim_overlay is not None:
                    stages["second_pass_residual_strip_detection"] = (
                        normalized.residual_trim_overlay.copy()
                    )
                stages["normalized_pdf_raster"] = detector_input.copy()
                stages["final_card_detector_input"] = detector_input.copy()
            diagnostics["stage_images"] = stages
        return CardSide(
            side=side,
            source_path=path,
            source_type=source_type,  # type: ignore[arg-type]
            source_page=source_page,
            original_image=detector_input.copy(),
            processed_image=detector_input.copy(),
            detector_input_image=detector_input,
            original_pdf_render=raw if source_type == "pdf" else None,
            normalized_pdf_raster=(
                detector_input.copy() if source_type == "pdf" else None
            ),
            source_diagnostics=diagnostics,
        )
