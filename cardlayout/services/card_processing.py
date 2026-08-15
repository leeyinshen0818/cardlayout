from __future__ import annotations

from cardlayout.models.card_side import CardSide
from cardlayout.models.detection import CardDetectionResult
from cardlayout.services.card_detector import CardDetector
from cardlayout.services.perspective_corrector import PerspectiveCorrector


class CardProcessingService:
    """Owns the non-destructive original → detected → processed transition."""

    def __init__(
        self,
        detector: CardDetector,
        perspective_corrector: PerspectiveCorrector | None = None,
    ) -> None:
        self.detector = detector
        self.perspective_corrector = perspective_corrector or PerspectiveCorrector(
            detector.card_size
        )

    def detect(self, card_side: CardSide) -> CardDetectionResult:
        processing_raster = card_side.processing_raster
        result = self.detector.detect(processing_raster)
        pipeline_diagnostics = self._pipeline_diagnostics(card_side)
        result.debug_info.update(pipeline_diagnostics)
        card_side.apply_detection(result)
        if result.success and len(result.polygon_points) == 4:
            inferred_count = int(result.debug_info.get("inferred_edge_count", 0))
            correction = self.perspective_corrector.correct(
                processing_raster,
                result.polygon_points,
                detector_confidence=result.confidence,
                inferred_corner_count=inferred_count,
                method="automatic",
                refine=True,
            )
            correction.debug_info.update(pipeline_diagnostics)
            card_side.apply_automatic_correction(correction)
        return result

    @staticmethod
    def _pipeline_diagnostics(card_side: CardSide) -> dict[str, object]:
        """Expose source metadata without allowing it to affect geometry logic."""
        source = card_side.source_diagnostics
        keys = (
            "source_type",
            "source_format",
            "normalized_width",
            "normalized_height",
            "pdf_trim_box",
            "pdf_residual_border_flags",
        )
        return {key: source.get(key) for key in keys}

    def apply_manual_correction(
        self,
        card_side: CardSide,
        points: tuple[tuple[float, float], ...],
    ):
        result = self.perspective_corrector.correct(
            card_side.processing_raster,
            points,
            detector_confidence=(
                card_side.detection_result.confidence
                if card_side.detection_result is not None
                else 0.0
            ),
            method="manual",
            refine=False,
        )
        if result.success:
            card_side.apply_manual_correction(result)
        return result

    @staticmethod
    def reset_correction(card_side: CardSide) -> None:
        card_side.reset_correction()

    @staticmethod
    def reset_user_edits(card_side: CardSide) -> None:
        card_side.reset_user_edits()

    @staticmethod
    def reset(card_side: CardSide) -> None:
        card_side.reset_detection()
