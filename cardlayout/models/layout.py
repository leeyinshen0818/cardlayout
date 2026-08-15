from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MmRect:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass(frozen=True, slots=True)
class LayoutPreset:
    name: str
    page_width_mm: float
    page_height_mm: float
    top_offset_mm: float
    vertical_gap_mm: float
    horizontal_alignment: str = "center"

    def __post_init__(self) -> None:
        if self.page_width_mm <= 0 or self.page_height_mm <= 0:
            raise ValueError("Page dimensions must be positive")
        if self.top_offset_mm < 0 or self.vertical_gap_mm < 0:
            raise ValueError("Layout spacing cannot be negative")
        if self.horizontal_alignment != "center":
            raise ValueError("Phase 1 supports centered cards only")


@dataclass(frozen=True, slots=True)
class PageLayout:
    page_width_mm: float
    page_height_mm: float
    front: MmRect
    back: MmRect


A4_LAYOUT = LayoutPreset(
    name="A4 portrait — stacked cards",
    page_width_mm=210.0,
    page_height_mm=297.0,
    top_offset_mm=51.6,
    vertical_gap_mm=20.0,
)
