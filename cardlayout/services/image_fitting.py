from __future__ import annotations

from PIL import Image, ImageOps


def to_rgb_on_white(image: Image.Image) -> Image.Image:
    """Normalize transparency and color mode for predictable print output."""
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    background.alpha_composite(rgba)
    return background.convert("RGB")


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Aspect-fit an image on white without cropping or distortion."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Target image size must be positive")
    fitted = ImageOps.contain(to_rgb_on_white(image), size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return canvas

