from __future__ import annotations

from pathlib import Path

from cardlayout.models.card_side import CardSide
from cardlayout.services.page_renderer import PageRenderer


class ExportError(OSError):
    pass


class ImageExporter:
    def __init__(self, renderer: PageRenderer, dpi: int = 300) -> None:
        self.renderer = renderer
        self.dpi = dpi

    def export(
        self,
        path: str | Path,
        front: CardSide | None,
        back: CardSide | None,
    ) -> None:
        destination = Path(path)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            page = self.renderer.render_page(front, back, self.dpi)
            page.save(
                destination,
                format="JPEG",
                quality=95,
                subsampling=0,
                dpi=(self.dpi, self.dpi),
            )
        except (OSError, ValueError) as exc:
            raise ExportError("The JPG could not be saved") from exc

