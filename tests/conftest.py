from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def sample_jpg(tmp_path: Path) -> Path:
    path = tmp_path / "front.jpg"
    Image.new("RGB", (428, 270), (190, 30, 45)).save(path, quality=95)
    return path


@pytest.fixture
def sample_png(tmp_path: Path) -> Path:
    path = tmp_path / "back.png"
    Image.new("RGBA", (428, 270), (20, 90, 190, 210)).save(path)
    return path


def make_pdf(path: Path, page_count: int) -> Path:
    import pymupdf as fitz

    document = fitz.open()
    for number in range(1, page_count + 1):
        page = document.new_page(width=300, height=200)
        page.draw_rect(page.rect, fill=(number / (page_count + 1), 0.2, 0.4))
        page.insert_text((30, 60), f"Page {number}", fontsize=24, color=(1, 1, 1))
    document.save(path)
    document.close()
    return path


@pytest.fixture
def one_page_pdf(tmp_path: Path) -> Path:
    return make_pdf(tmp_path / "one.pdf", 1)


@pytest.fixture
def two_page_pdf(tmp_path: Path) -> Path:
    return make_pdf(tmp_path / "two.pdf", 2)


@pytest.fixture
def three_page_pdf(tmp_path: Path) -> Path:
    return make_pdf(tmp_path / "three.pdf", 3)
