from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CardSizePreset:
    name: str
    width_mm: float
    height_mm: float

    def __post_init__(self) -> None:
        if self.width_mm <= 0 or self.height_mm <= 0:
            raise ValueError("Card dimensions must be positive")

    @property
    def label(self) -> str:
        return f"{self.name} — {self.width_mm:g} × {self.height_mm:g} mm"


MALAYSIA_IC = CardSizePreset("Malaysia IC", 85.6, 54.0)

