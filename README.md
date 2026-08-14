# CardLayout

CardLayout is a local desktop application for placing the front and back of a physical card on an A4 portrait page and exporting a print-ready PDF or JPG. It is a general card layout tool; Malaysia IC is simply the first built-in size preset.

## Current capabilities (Phases 1–2)

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
- Automatically attempt physical-card detection after every image or PDF-page import.
- Crop surrounding background and correct ordinary in-plane rotation when confidence is acceptable.
- Use a staged hybrid detector: fast contours first, higher-resolution color/line analysis when needed, then occlusion-aware reconstruction as a fallback.
- Generate and fuse contour, rotated-rectangle, Hough-line, three-edge, and two-opposite-edge candidates across working scales.
- Rank candidates using geometry, preset aspect ratio, edge and parallel-line support, interior structure, area, rectangularity, cross-method agreement, inferred-edge cost, and ambiguity rather than choosing the largest contour.
- Fail safely to the original image when detection confidence is low.
- Compare Original and Detected previews, re-run detection, or reset a side to its untouched original.

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

Tests cover JPG/PNG normalization, one/two/multi-page PDFs, invalid input, A4/card geometry, PDF/JPG export, consistent placement, detection scoring and geometry, rotated cards, clutter, ambiguous candidates, safe failure, processing state, and UI interactions. Tests do not require manually opening the GUI.

## Architecture

```text
cardlayout/
  models/       CardSide, detection result/configuration, size and A4 layout models
  services/     input/PDF loading, OpenCV detection, rendering, exporters
  ui/           PySide6 input widgets, comparison controls, A4 preview
tests/          input, detection, processing, layout, export, and UI tests
```

`CardSide.original_image` always preserves the normalized input. A successful `CardDetectionResult` stores confidence, bounding geometry, polygon points, rotation, candidate method, tuning metrics, and a separate `detected_image`; only then does that image become `processed_image` for layout and export. Resetting never reopens or recompresses the source. Future perspective correction can consume the retained polygon without changing the import or layout layers.

Detection normally starts at a 1100-pixel long edge. Weak or ambiguous cases retry at up to 1800 pixels with LAB/HSV color edges and line detection, then up to 2400 pixels for partial-edge reconstruction. Geometry is mapped back to the original resolution, and the final rotated crop is made from the full-resolution source. All scales, thresholds, weights, penalties, and confidence cutoffs live in `CardDetectionConfig` rather than being scattered through the detector.

Optional debug mode retains original/working images, edge and threshold maps, Hough segments, candidate polygons, inferred-edge counts, candidate scores, the winner and runner-up, confidence reasoning, scales, and processing time in memory only. Structured debug logging contains numeric detection diagnostics but no image content or personal data.

## Not implemented yet

Full perspective/homography correction, draggable corner editing, background segmentation/removal, advanced image enhancement, sharpening/restoration, OCR, and AI features are not implemented. Phase 2 performs surrounding-background cropping and basic in-plane rotation only.
