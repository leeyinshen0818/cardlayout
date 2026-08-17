from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


OrientationAction = Literal[
    "flip_horizontal",
    "flip_vertical",
    "rotate_180",
    "reset",
]


@dataclass(frozen=True, slots=True)
class OrientationState:
    """Non-destructive orientation choices for one card side."""

    flip_horizontal: bool = False
    flip_vertical: bool = False

    @property
    def is_normal(self) -> bool:
        return not self.flip_horizontal and not self.flip_vertical

    @property
    def label(self) -> str:
        if self.flip_horizontal and self.flip_vertical:
            return "Rotated 180°"
        if self.flip_horizontal:
            return "Flipped Horizontal"
        if self.flip_vertical:
            return "Flipped Vertical"
        return "Normal"

    def apply_action(self, action: OrientationAction) -> OrientationState:
        if action == "flip_horizontal":
            return replace(self, flip_horizontal=not self.flip_horizontal)
        if action == "flip_vertical":
            return replace(self, flip_vertical=not self.flip_vertical)
        if action == "rotate_180":
            return replace(
                self,
                flip_horizontal=not self.flip_horizontal,
                flip_vertical=not self.flip_vertical,
            )
        if action == "reset":
            return OrientationState()
        raise ValueError(f"Unknown orientation action: {action}")


@dataclass(frozen=True, slots=True)
class AutomaticOrientationResult:
    """Result of the conservative post-perspective orientation assessment."""

    recommended_state: OrientationState = OrientationState()
    confidence: float = 0.0
    applied: bool = False
    method: str = "local_text_orientation_unavailable"
    reason: str = (
        "No reliable local text-orientation recognizer is bundled; image unchanged."
    )
