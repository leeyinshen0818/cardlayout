from __future__ import annotations

from cardlayout.models.card_side import SideName
from cardlayout.models.card_size import CardSizePreset
from cardlayout.models.layout import LayoutPreset, MmRect, PageLayout

MM_PER_INCH = 25.4
PDF_POINTS_PER_INCH = 72.0


class LayoutEngine:
    """Single source of truth for preview and exported page geometry."""

    def __init__(self, layout: LayoutPreset, card_size: CardSizePreset) -> None:
        self.preset = layout
        self.card_size = card_size
        self._vertical_offsets_mm: dict[SideName, float] = {
            "front": 0.0,
            "back": 0.0,
        }

    def calculate(self) -> PageLayout:
        x = (self.preset.page_width_mm - self.card_size.width_mm) / 2.0
        front = MmRect(
            x=x,
            y=self.preset.top_offset_mm + self._vertical_offsets_mm["front"],
            width=self.card_size.width_mm,
            height=self.card_size.height_mm,
        )
        base_back_y = (
            self.preset.top_offset_mm
            + self.card_size.height_mm
            + self.preset.vertical_gap_mm
        )
        back = MmRect(
            x=x,
            y=base_back_y + self._vertical_offsets_mm["back"],
            width=self.card_size.width_mm,
            height=self.card_size.height_mm,
        )
        if front.y < 0 or front.bottom > self.preset.page_height_mm:
            raise ValueError("The Front position is outside the page")
        if back.y < 0 or back.bottom > self.preset.page_height_mm:
            raise ValueError("The Back position is outside the page")
        return PageLayout(
            page_width_mm=self.preset.page_width_mm,
            page_height_mm=self.preset.page_height_mm,
            front=front,
            back=back,
        )

    def vertical_offset(self, side: SideName) -> float:
        return self._vertical_offsets_mm[side]

    def vertical_offset_limits(self, side: SideName) -> tuple[float, float]:
        if side == "front":
            base_y = self.preset.top_offset_mm
        else:
            base_y = (
                self.preset.top_offset_mm
                + self.card_size.height_mm
                + self.preset.vertical_gap_mm
            )
        return (-base_y, self.preset.page_height_mm - self.card_size.height_mm - base_y)

    def set_vertical_offset(self, side: SideName, offset_mm: float) -> float:
        minimum, maximum = self.vertical_offset_limits(side)
        if not minimum <= offset_mm <= maximum:
            raise ValueError(f"{side.title()} must stay within the A4 page")
        self._vertical_offsets_mm[side] = round(float(offset_mm), 3)
        return self._vertical_offsets_mm[side]

    def adjust_vertical_offset(self, side: SideName, delta_mm: float) -> float:
        minimum, maximum = self.vertical_offset_limits(side)
        requested = self._vertical_offsets_mm[side] + delta_mm
        return self.set_vertical_offset(side, min(max(requested, minimum), maximum))

    def reset_vertical_offset(self, side: SideName) -> None:
        self._vertical_offsets_mm[side] = 0.0

    @staticmethod
    def mm_to_pixels(value_mm: float, dpi: float) -> float:
        return value_mm * dpi / MM_PER_INCH

    @staticmethod
    def mm_to_points(value_mm: float) -> float:
        return value_mm * PDF_POINTS_PER_INCH / MM_PER_INCH

    @classmethod
    def rect_at_dpi(cls, rect: MmRect, dpi: float) -> tuple[int, int, int, int]:
        return tuple(
            int(round(cls.mm_to_pixels(value, dpi)))
            for value in (rect.x, rect.y, rect.right, rect.bottom)
        )

    @classmethod
    def rect_in_points(cls, rect: MmRect) -> tuple[float, float, float, float]:
        return tuple(
            cls.mm_to_points(value)
            for value in (rect.x, rect.y, rect.right, rect.bottom)
        )

    def page_pixels(self, dpi: float) -> tuple[int, int]:
        page = self.calculate()
        return (
            int(round(self.mm_to_pixels(page.page_width_mm, dpi))),
            int(round(self.mm_to_pixels(page.page_height_mm, dpi))),
        )

    def page_points(self) -> tuple[float, float]:
        page = self.calculate()
        return (
            self.mm_to_points(page.page_width_mm),
            self.mm_to_points(page.page_height_mm),
        )
