from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from PIL import Image

SideName = Literal["front", "back"]
SourceType = Literal["image", "pdf"]


@dataclass(slots=True)
class CardSide:
    """One normalized card face, independent of its original file format."""

    side: SideName
    source_path: Path
    source_type: SourceType
    source_page: int | None
    original_image: Image.Image
    processed_image: Image.Image

    @property
    def display_name(self) -> str:
        if self.source_type == "pdf" and self.source_page is not None:
            return f"{self.source_path.name} — page {self.source_page}"
        return self.source_path.name

    def assigned_to(self, side: SideName) -> CardSide:
        """Return a new assignment without reloading or copying image pixels."""
        return replace(self, side=side)

