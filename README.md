# CardLayout

CardLayout is a privacy-friendly Windows desktop application for preparing the
front and back of a physical card for accurate A4 printing. Import images or PDF
pages, correct the card boundaries, preview the physical layout, and export a
print-ready PDF or 300-DPI JPG.

Everything runs locally. CardLayout does not upload documents, perform OCR, or
require an internet connection.

## Highlights

- Import JPG, JPEG, PNG, and PDF sources.
- Open a two-page PDF as Front and Back in one step.
- Automatically detect the physical card boundary and correct perspective.
- Handle low-texture card backs, weak edges, rounded corners, internal printed
  rectangles, and partial hand occlusion conservatively.
- Adjust all four corners manually when automatic detection needs review.
- Apply independent sharpening and brightness/contrast presets to Front and Back.
- Move Front and Back vertically in precise 1 mm steps.
- Preview the complete A4 portrait page at the correct relative dimensions.
- Export an A4 PDF or a 300-DPI JPG directly to Downloads.
- Run as a single standalone `CardLayout.exe` with no Python installation.
- Start maximized and automatically use a compact layout on smaller or
  display-scaled laptops.

## Typical workflow

1. Open a two-page PDF, choose separate Front and Back files, or drag supported
   files into the window.
2. Review the automatically detected and perspective-corrected card images.
3. Select **Adjust Corners** if a physical boundary needs manual correction.
4. Click either card preview to open its Corrections sidebar or position controls.
5. Review the A4 preview.
6. Select **Export PDF** or **Export JPG · 300 DPI**.

Exports are saved automatically in the Windows Downloads folder. Existing files
are never overwritten; CardLayout creates names such as `card-layout (1).pdf`.
After export, Windows Explorer opens the containing folder automatically.

## Input behavior

CardLayout supports:

- Individual `.jpg`, `.jpeg`, `.png`, and `.pdf` files for either side
- One two-page PDF, mapped as Page 1 → Front and Page 2 → Back
- One-page PDFs, which leave Back empty
- PDFs with more than two pages, using only the first two pages with a notice
- Drag and drop of one PDF or up to two supported files

PDF pages are rendered at high quality before entering the shared raster pipeline.
For PDF input only, border-connected white, off-white, cream, or light-gray page
whitespace can be removed before card detection. A safety margin protects rounded
corners, shadows, and weak card edges. JPG and PNG files do not receive this
PDF-specific trimming.

## Detection and correction

CardLayout uses a geometry-first OpenCV pipeline. It prioritizes the expected
outer perimeter, long-edge support, the selected card-size ratio, and consistency
with the rough detected region. Text, portraits, chips, logos, holograms, and
internal rectangles are not treated as dominant evidence.

The top, right, bottom, and left edge confidences are tracked independently. A
weak or partly covered edge lowers confidence instead of forcing detection onto a
smaller internal rectangle. Suspicious refinements are rejected when they shrink,
expand, shift, or distort the rough card geometry without strong boundary support.

The image used for preview and export follows this safe fallback order:

1. Valid manual corner correction
2. Validated automatic perspective correction
3. Rough detected card crop
4. Original normalized image

A questionable refinement never replaces a usable rough detection. CardLayout
shows a review recommendation when it cannot confidently recover the full physical
boundary.

### Manual corners

The corner editor provides four labeled handles, connecting lines, a live
rectified preview, zoom, pan, Fit, 100%, reset-to-automatic, Apply, and Cancel.
Invalid crossing, overlapping, non-convex, out-of-bounds, or near-zero-area corner
arrangements cannot be applied.

### Image corrections

The collapsible right sidebar provides thumbnail previews for:

- Sharpen / Soften: Soft, Normal, Sharp, Sharper
- Brightness / Contrast: Normal, Bright +10, Bright +20,
  Bright + Contrast, Strong Bright + Contrast

Front and Back keep independent settings. **Reset corrections** restores only the
selected side's appearance. The side-level **Reset** restores automatic geometry
and Normal appearance without reloading the source file.

Opening or closing the sidebar only changes the UI viewport. It does not rerun
detection, reload images, reset corners, or change physical A4 coordinates.

## A4 layout and exports

The built-in card preset is **Malaysia IC — 85.6 × 54 mm**. Both sides are
centered horizontally on an A4 portrait page.

| Setting | Value |
| --- | ---: |
| A4 page | 210 × 297 mm |
| Front top position | 51.6 mm |
| Back top position | 125.6 mm |
| Gap between cards | 20 mm |
| JPG output | 2480 × 3508 px at 300 DPI |

Front and Back can each be moved up or down independently in 1 mm steps and reset
to their default position. These adjustments change physical placement; opening
the Corrections sidebar or resizing the window does not.

For correctly sized output, print the exported PDF using **Actual Size** or
**100%**. Do not select Fit, Shrink, or Scale to Page.

## Standalone Windows app

The packaged application is one file:

```text
dist\CardLayout.exe
```

Copy only `CardLayout.exe` to the Desktop or any other folder on another Windows
computer and double-click it. The target computer does not need Python, pip,
VS Code, or separate dependencies.

The executable:

- Uses the bundled CardLayout icon in Explorer, the app window, and the taskbar
- Opens without a console window
- Starts maximized
- Bundles PySide6, OpenCV, NumPy, Pillow, PyMuPDF, and required Qt resources
- Excludes the `tests` and `testing` directories

Because this is a PyInstaller one-file build, the first launch can take a little
longer while bundled files are extracted to a temporary directory. That temporary
runtime is cleaned up when the application closes.

## Responsive UI

CardLayout chooses its interface density from the available Qt logical screen
size, which accounts for Windows display scaling. Common laptop workspaces such
as 1600 × 900, 1536 × 864, and 1280 × 720 automatically use a compact layout.
Larger desktop displays retain the standard spacing.

Compact mode reduces panel width, preview height, fonts, button size, margins, and
spacing while keeping the same controls and physical A4 placement.

## Run from source

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m cardlayout
```

The installed command can also launch the application:

```powershell
cardlayout
```

## Build the standalone EXE

Build on Windows with PyInstaller:

```powershell
python -m pip install -e ".[dev]" pyinstaller
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1
```

The completed single-file application is written to:

```text
dist\CardLayout.exe
```

The PyInstaller configuration is stored in `packaging\CardLayout.spec`. Resource
paths are resolved for both source execution and PyInstaller's temporary one-file
runtime, so the build does not depend on development-machine absolute paths.

## Tests

Run the complete test suite with:

```powershell
python -m pytest
```

The tests cover input normalization, PDF/JPG parity, PDF frame trimming, layout
geometry, exporters, detection scoring, low-texture backs, nested rectangles,
partial edge and corner occlusion, perspective refinement, collapse protection,
manual corners, appearance corrections, reset behavior, responsive UI, and
sidebar interactions. GUI tests run without manually opening the application.

## Privacy

- All image and PDF processing happens locally.
- No source files or document contents are uploaded.
- No OCR, face recognition, template matching, cloud service, or AI reconstruction
  is used.
- PDF pages are rendered in memory; hidden page-image files are not created.
- Detection diagnostics contain numeric geometry and confidence data, not document
  image content.

## Project structure

```text
cardlayout/
  models/       Card, detection, correction, size, and A4 layout state
  services/     Input, normalization, detection, rectification, and export
  ui/           Main window, card widgets, corner editor, and A4 preview
icon/            Application icon
packaging/       PyInstaller specification
tests/           Automated regression tests
build_windows.ps1
```

`CardSide.original_image` preserves the normalized source. Detection, automatic
perspective correction, manual correction, and appearance settings remain separate
states. `CardSide.best_image` is the single source used by the side previews, A4
preview, PDF exporter, and JPG exporter.

## Current scope

CardLayout does not include background removal, white balance, saturation controls,
restoration, denoise, deblur, OCR, orientation recognition, or AI-based image
reconstruction. Appearance presets adjust existing pixels and cannot restore
missing detail.
