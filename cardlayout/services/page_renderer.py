from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from cardlayout.models.card_side import CardSide
from cardlayout.models.layout import MmRect
from cardlayout.services.image_fitting import fit_image
from cardlayout.services.layout_engine import LayoutEngine


class PageRenderer:
    """Rasterizes the logical layout; used by JPG and PDF card artwork."""

    BORDER_COLOR = (172, 181, 194)
    PLACEHOLDER_TEXT = (113, 124, 140)

    def __init__(self, engine: LayoutEngine) -> None:
        self.engine = engine

    def render_card(
        self,
        side: CardSide | None,
        size: tuple[int, int],
        label: str,
        border_width: int = 2,
    ) -> Image.Image:
        if side is not None:
            card = fit_image(side.best_image, size)
        else:
            card = Image.new("RGB", size, (247, 249, 252))
            draw = ImageDraw.Draw(card)
            font = ImageFont.load_default()
            box = draw.textbbox((0, 0), label, font=font)
            text_width = box[2] - box[0]
            text_height = box[3] - box[1]
            draw.text(
                ((size[0] - text_width) / 2, (size[1] - text_height) / 2),
                label,
                fill=self.PLACEHOLDER_TEXT,
                font=font,
            )

        draw = ImageDraw.Draw(card)
        for offset in range(max(1, border_width)):
            draw.rectangle(
                (offset, offset, size[0] - 1 - offset, size[1] - 1 - offset),
                outline=self.BORDER_COLOR,
            )
        return card

    def render_page(
        self,
        front: CardSide | None,
        back: CardSide | None,
        dpi: float,
    ) -> Image.Image:
        page_layout = self.engine.calculate()
        page = Image.new("RGB", self.engine.page_pixels(dpi), "white")
        if front is not None:
            self._paste_side(page, front, page_layout.front, dpi, "FRONT")
        if back is not None:
            self._paste_side(page, back, page_layout.back, dpi, "BACK")
        return page

    def _paste_side(
        self,
        page: Image.Image,
        side: CardSide | None,
        rect: MmRect,
        dpi: float,
        label: str,
    ) -> None:
        left, top, right, bottom = self.engine.rect_at_dpi(rect, dpi)
        card = self.render_card(
            side,
            (right - left, bottom - top),
            label,
            border_width=max(1, int(round(dpi / 150))),
        )
        page.paste(card, (left, top))
