from __future__ import annotations

from PIL import Image, ImageEnhance, ImageFilter

from cardlayout.models.image_correction import (
    ImageCorrectionState,
    SHARPEN_PRESETS,
    TONE_PRESETS,
)


def apply_image_correction(
    source: Image.Image, state: ImageCorrectionState
) -> Image.Image:
    """Render one correction state directly from an unchanged geometry source."""
    if state.is_normal:
        return source.copy()

    image = source.convert("RGB")
    tone = TONE_PRESETS[state.tone]
    if tone.brightness != 1.0:
        image = ImageEnhance.Brightness(image).enhance(tone.brightness)
    if tone.contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(tone.contrast)

    sharpen = SHARPEN_PRESETS[state.sharpen]
    if sharpen.blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(sharpen.blur_radius))
    elif sharpen.unsharp_percent > 0:
        image = image.filter(
            ImageFilter.UnsharpMask(
                radius=sharpen.unsharp_radius,
                percent=sharpen.unsharp_percent,
                threshold=sharpen.unsharp_threshold,
            )
        )
    return image.copy()


def correction_thumbnail(
    source: Image.Image,
    state: ImageCorrectionState,
    size: tuple[int, int] = (88, 56),
) -> Image.Image:
    """Apply a preset to a small copy, never the full-resolution source."""
    preview = source.convert("RGB").copy()
    preview.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (244, 246, 249))
    left = (size[0] - preview.width) // 2
    top = (size[1] - preview.height) // 2
    canvas.paste(preview, (left, top))
    return apply_image_correction(canvas, state)
