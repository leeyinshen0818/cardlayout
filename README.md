# CardLayout

CardLayout is a local desktop application for placing the front and back of a physical card on an A4 portrait page and exporting a print-ready PDF or JPG. It is a general card layout tool; Malaysia IC is simply the first built-in size preset.

## Phase 1 capabilities

- Load Front and Back from JPG, JPEG, or PNG files.
- Load Page 1 from separate PDF files, including mixed image/PDF input.
- Open one PDF and map Page 1 to Front and Page 2 to Back. One-page PDFs leave Back empty; PDFs with more pages clearly report that only the first two are used.
- Swap or clear either side without reloading source files.
- Click Front or Back in the A4 preview to reveal floating controls for moving that side independently in 1 mm steps or resetting its position.
- Preview the complete A4 portrait layout live.
- Aspect-fit source images without stretching or cropping; unused space is letterboxed.
- Export a physical A4 PDF intended for printing at **Actual Size / 100%**.
- Export the full A4 page as a 300-DPI JPG (2480 × 3508 pixels).
- Drag one PDF or up to two supported files onto the application.

The current preset is **Malaysia IC — 85.6 × 54 mm**. Both cards are centered horizontally. The default pair is shifted 25 mm lower than the original Phase 1 position: Front starts 61.6 mm from the page top and Back starts at 135.6 mm, leaving a 20 mm gap. Each side can then be moved independently while remaining inside the A4 page.

## Privacy

All file processing is local. CardLayout does not upload files, call web services, perform OCR, or log document content. PDF pages are rendered directly in memory, so no hidden page-image copies are produced.

## Install and run

Python 3.10 or newer is required. From the project directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m cardlayout
```

The installed console command `cardlayout` can also launch the app.

## Development and tests

Install the development dependencies as shown above, then run:

```powershell
python -m pytest
```

Tests cover JPG/PNG normalization, one/two/multi-page PDFs, invalid input, A4/card geometry, PDF page dimensions, 300-DPI JPG dimensions, and consistent placement between output formats. Tests do not require opening the GUI.

## Architecture

```text
cardlayout/
  models/       CardSide, card-size presets, logical millimetre layout
  services/     input/PDF loading, aspect fitting, shared rendering, exporters
  ui/           PySide6 input widgets, live A4 preview, main window
tests/          non-UI input, layout, and export tests
```

`CardSide.original_image` preserves the normalized input while `processed_image` feeds the layout. Future processing stages can replace the latter without changing preview or export code. `LayoutEngine` is the geometry source shared by the preview and both exporters.

## Not implemented yet

Phase 1 intentionally does **not** include automatic card detection, background detection or removal, perspective correction, image cleanup/enhancement, OCR, or AI features.
