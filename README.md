# CardLayout

CardLayout is a local desktop application for placing the front and back of a physical card on an A4 portrait page and exporting a print-ready PDF or JPG. It is a general card layout tool; Malaysia IC is simply the first built-in size preset.

## Current capabilities (Phases 1–3.1)

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
- Automatically rectify reliable detected quadrilaterals into a front-facing landscape card at the active preset's exact aspect ratio.
- Refine rough detector geometry at high resolution inside a padded local card ROI before rectification.
- Fit the top, right, bottom, and left physical boundaries independently with direction-constrained, distributed-support RANSAC and a narrower second precision pass.
- Combine grayscale/CLAHE, LAB, HSV, Canny, Sobel magnitude, and gradient-direction evidence without changing final image pixels.
- Handle rounded corners by intersecting fitted straight boundary lines instead of snapping to curved contour tips.
- Retain weak rough edges and lower confidence when physical boundary support is incomplete or partially occluded.
- Preserve source quality by applying one perspective transform from original-resolution pixels, with a configurable 1200-pixel preferred width and conservative upscale limit.
- Compare Original, Detected, and Corrected previews, re-run detection, reset correction, or reset a side to its untouched original.
- Adjust all four source corners manually in a dedicated editor with labeled handles, connecting lines, live rectified preview, mouse-wheel zoom, pan, Fit, 100%, reset-to-automatic, Apply, and Cancel.
- Reject overlapping, crossing, out-of-bounds, non-convex, or near-zero-area manual corner geometry before it can be applied.
- Keep manual corrections authoritative until the user resets correction, explicitly accepts re-detection, or imports a new source.
- Use the same best-stage priority for A4 preview, PDF, and JPG: manual correction, automatic correction, detected crop, then original.

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

Tests cover JPG/PNG normalization, one/two/multi-page PDFs, invalid input, A4/card geometry, PDF/JPG export, consistent placement, detection scoring and geometry, rotated cards, clutter, ambiguous candidates, safe failure, perspective ordering and validation, robust line fitting and outlier rejection, local search bands, rounded and partially occluded boundaries, color-only edges, strong internal rectangles, refinement fallback, exact preset ratios, correction state, corrected-stage export, draggable corner controls, zoom/pan coordinate stability, and UI interactions. Tests do not require manually opening the GUI.

## Architecture

```text
cardlayout/
  models/       CardSide, detection/perspective state, size and A4 layout models
  services/     loading, OpenCV detection/rectification, rendering, exporters
  ui/           input widgets, four-corner editor, comparison and A4 previews
tests/          input, detection, perspective, state, layout, export, and UI tests
```

`CardSide.original_image` always preserves the normalized input. A successful `CardDetectionResult` stores confidence, bounding geometry, polygon points, rotation, candidate method, tuning metrics, and a separate `detected_image`. `PerspectiveResult` separately stores ordered/refined source corners, destination geometry, confidence, warnings, output dimensions, transform data, and `rectified_image`. Automatic and manual results remain distinct; `CardSide.best_image` is the single source of truth for layout and export. Resetting never reopens or recompresses the source.

Detection normally starts at a 1100-pixel long edge. Weak or ambiguous cases retry at up to 1800 pixels with LAB/HSV color edges and line detection, then up to 2400 pixels for partial-edge reconstruction. Geometry is mapped back to the original resolution, and the final rotated crop is made from the full-resolution source. All scales, thresholds, weights, penalties, and confidence cutoffs live in `CardDetectionConfig` rather than being scattered through the detector.

Optional debug mode retains original/working images, edge and threshold maps, Hough segments, candidate polygons, inferred-edge counts, candidate scores, the winner and runner-up, confidence reasoning, scales, and processing time in memory only. Structured debug logging contains numeric detection diagnostics but no image content or personal data.

Perspective correction consumes the detector's original-image polygon instead of re-detecting. Corners are normalized to top-left, top-right, bottom-right, and bottom-left; physical edges are refined as described below, validated, and transformed once from the original-resolution image. Medium-confidence detection, inferred geometry, or image-boundary clipping remains usable but is labeled **Review correction**.

Phase 3.1 treats detector corners as rough priors. It pads the detected card by 10%, caps exceptionally large refinement ROIs at a high-resolution 2600-pixel long edge, and searches only narrow bands around each expected boundary. The first pass tolerates rough detector error; the second pass is centered on the initial fit. RANSAC hypotheses must have spatially distributed gradient support, plausible orientation, and proximity to the rough edge. Final corners are the intersections of the four infinite fitted lines. Excessive displacement, invalid geometry, or insufficient evidence falls back to stored rough corners and requests review.

Perspective debug mode additionally exposes rough geometry, search bands, raw edge evidence, inliers and rejected outliers, fitted physical lines, final intersections, per-edge scores/support/residuals, per-corner confidence, rough-to-refined displacement, refinement confidence, and fallback reason. Images remain in memory unless a developer explicitly saves them.

## Not implemented yet

Background segmentation/removal, automatic brightness or white-balance correction, contrast enhancement, sharpening/restoration, denoise, deblur, OCR, upside-down content recognition, and AI features are not implemented. Phase 3 focuses only on geometry.
