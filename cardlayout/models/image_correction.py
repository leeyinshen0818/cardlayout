from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

SharpenPresetId = Literal["soft", "normal", "sharp", "sharper"]
TonePresetId = Literal[
    "normal",
    "bright_10",
    "bright_20",
    "bright_contrast",
    "strong_bright_contrast",
]


@dataclass(frozen=True, slots=True)
class SharpenPreset:
    key: SharpenPresetId
    label: str
    blur_radius: float = 0.0
    unsharp_radius: float = 0.0
    unsharp_percent: int = 0
    unsharp_threshold: int = 0


@dataclass(frozen=True, slots=True)
class TonePreset:
    key: TonePresetId
    label: str
    brightness: float = 1.0
    contrast: float = 1.0


SHARPEN_PRESETS: dict[SharpenPresetId, SharpenPreset] = {
    "soft": SharpenPreset("soft", "Soft", blur_radius=0.65),
    "normal": SharpenPreset("normal", "Normal"),
    "sharp": SharpenPreset(
        "sharp", "Sharp", unsharp_radius=1.2, unsharp_percent=70, unsharp_threshold=3
    ),
    "sharper": SharpenPreset(
        "sharper",
        "Sharper",
        unsharp_radius=1.4,
        unsharp_percent=115,
        unsharp_threshold=3,
    ),
}

TONE_PRESETS: dict[TonePresetId, TonePreset] = {
    "normal": TonePreset("normal", "Normal"),
    "bright_10": TonePreset("bright_10", "Bright +10", brightness=1.10),
    "bright_20": TonePreset("bright_20", "Bright +20", brightness=1.20),
    "bright_contrast": TonePreset(
        "bright_contrast", "Bright + Contrast", brightness=1.12, contrast=1.10
    ),
    "strong_bright_contrast": TonePreset(
        "strong_bright_contrast",
        "Strong Bright + Contrast",
        brightness=1.20,
        contrast=1.15,
    ),
}


@dataclass(frozen=True, slots=True)
class ImageCorrectionState:
    """Independent, non-destructive manual correction choices for one card side."""

    sharpen: SharpenPresetId = "normal"
    tone: TonePresetId = "normal"

    def __post_init__(self) -> None:
        if self.sharpen not in SHARPEN_PRESETS:
            raise ValueError(f"Unknown sharpen preset: {self.sharpen}")
        if self.tone not in TONE_PRESETS:
            raise ValueError(f"Unknown brightness/contrast preset: {self.tone}")

    @property
    def is_normal(self) -> bool:
        return self.sharpen == "normal" and self.tone == "normal"

    def with_sharpen(self, preset: SharpenPresetId) -> ImageCorrectionState:
        return replace(self, sharpen=preset)

    def with_tone(self, preset: TonePresetId) -> ImageCorrectionState:
        return replace(self, tone=preset)

