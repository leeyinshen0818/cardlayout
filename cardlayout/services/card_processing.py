from __future__ import annotations

from cardlayout.models.card_side import CardSide
from cardlayout.models.detection import CardDetectionResult
from cardlayout.services.card_detector import CardDetector


class CardProcessingService:
    """Owns the non-destructive original → detected → processed transition."""

    def __init__(self, detector: CardDetector) -> None:
        self.detector = detector

    def detect(self, card_side: CardSide) -> CardDetectionResult:
        result = self.detector.detect(card_side.original_image)
        card_side.apply_detection(result)
        return result

    @staticmethod
    def reset(card_side: CardSide) -> None:
        card_side.reset_detection()
