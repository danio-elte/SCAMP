# SCAMP — Zebrafish Spine Straightener & Heatmap

SCAMP is a single-file, napari-style GUI for analysing Alizarin Red S–stained zebrafish spinal mineralization.

It now covers the complete workflow from raw CZI stacks to background-subtracted projections, straightened spine images, quality-control reports, cohort heatmaps, and output tables.

---

## What SCAMP replaces

| Previous script | Previous role | Current SCAMP role |
|---|---|---|
| `norm_mean_IP.py` | CZI + ROI background subtraction and projection | `Subtract background` |
| `interactive_straighten2.py` | manual guide placement and straightening | `Add straighten guides` + `Straighten current` |
| `gerinc_illesztes_batch_heatmap.py` | profile extraction, heatmap, tables | `Process all - heatmap + tables` |
| `warp.py` | straightening helper | replaced by SCAMP ribbon straightening |

---

## Install

```bash
conda env create -f spine_env.yaml
conda activate spine
```

If you are updating an existing environment:

```bash
conda env update -f spine_env.yaml
conda activate spine
```

---

## Run

```bash
python scamp.py
```

or, if you are using the dark UI build:

```bash
python scamp_darkly_clean_qc.py
```

---

## Recommended workflow

```text
Open CZI
→ Load preview / edit ROI
→ Transform if needed
→ Rectangle tool
→ Save ROI
→ Close preview if desired
→ Subtract background
→ Add straighten guides
→ Set guides
→ Straighten current
→ Process all - heatmap + tables
```

---

## 1. Open CZI files

Use:

```text
Open
```

CZI files are loaded lazily. This means opening many files does not immediately load every full stack into memory.

Each CZI appears in the Open files panel as a registered file. To inspect or edit one file, click:

```text
Load preview / edit ROI
```

This loads only that CZI into a temporary preview tab.

After ROI or geometry editing, the preview tab can be closed. The CZI remains registered, and saved ROI/geometry data remain available for batch processing.

---

## 2. Preview and Z browsing

When a CZI preview is loaded, SCAMP displays:

- a Z-slider for browsing the stack
- a Tools menu
- ROI tools
- transform tools
- projection tools

Preview tabs can be closed to free memory. This is recommended when working through many CZI files.

---

## 3. Transform before ROI if needed

If the animal orientation needs correction before ROI placement, use:

```text
Tools → Flip horizontal
Tools → Flip vertical
Tools → Rotate with preview…
Tools → Crop to rectangle
```

SCAMP stores these geometry operations in a sidecar file:

```text
<base>_scamp_geometry.json
```

When `Subtract background` is later run, SCAMP reloads the original CZI and reapplies the saved geometry before applying the ROI. This ensures the batch result matches the edited preview geometry.

---

## 4. Background ROI

Use:

```text
Tools → Rectangle tool
```

The ROI is used to estimate background.

Rules:

- only one ROI is active at a time
- dragging inside the ROI moves it
- clicking outside replaces it
- tiny accidental selections are ignored

Recommended ROI placement:

- use a clean background-like region near the specimen
- avoid mineralized vertebrae
- avoid bright artifacts
- avoid very dark vignetted corners if they do not represent the specimen background

---

## 5. Save ROI

Use:

```text
Tools → Save ROI
```

SCAMP saves an ImageJ-compatible `.roi` file next to the original CZI:

```text
A1.czi
A1.roi
```

The ROI is also shown under the matching CZI in the Open files panel.

---

## 6. Subtract background

Use the fixed button below the Open files list:

```text
Subtract background
```

A dialog appears before processing. The default estimator is:

```text
Median
```

Available background estimators:

- Median
- Mean
- Percentile

Median is recommended as the default because it is robust to occasional bright pixels inside the background ROI.

For every CZI with a matching ROI, SCAMP:

1. loads the CZI stack
2. reapplies saved geometry
3. reads the matching ROI
4. estimates background per Z slice
5. subtracts background from each Z slice
6. clips negative values to zero
7. creates a Z-depth-normalised sum projection
8. releases the CZI stack from memory

---

## 7. Z-depth-normalised sum projection

The projection is calculated as:

```text
sum(background-corrected stack) / total Z depth in µm
```

where:

```text
total Z depth = number of Z slices × Z-step size
```

Outputs are written to:

```text
background_subtracted/
```

---

## 8. Storage scale and 16-bit TIFF output

All generated scientific image outputs are saved as 16-bit TIFF.

SCAMP uses a fixed global storage scale for Z-depth-normalised quantitative images:

```text
storage scale = ×1000
```

This is not biological normalisation and does not change sample-to-sample ratios.

It only prevents low float values such as:

```text
0.03, 0.42, 0.87
```

from being converted to zero during `float → uint16` TIFF saving.

SCAMP stores the values as:

```text
30, 420, 870
```

and records the storage scale in QC metadata.

The same storage-unit policy is used consistently for:

- background-subtracted projections
- straightened images
- Process all input data

This prevents weak background or low fluorescence values from being lost during saving and reloading.

---

## 9. Add straighten guides

Projection images open without straighten guides.

Select a projection image and click:

```text
Add straighten guides
```

This adds the standard guide triplets.

Each triplet contains:

- green point = midline
- red point = dorsal boundary
- blue point = ventral boundary

The dorsal–ventral guide distance is fixed. Dragging red/blue points changes angle, not length.

---

## 10. Straighten current

Use:

```text
Straighten current
```

SCAMP now uses dense ribbon straightening instead of triangulation-based piecewise-affine filling.

This means:

- every output pixel is sampled directly from the guide-defined dorsal–ventral corridor
- internal black holes from triangulation gaps are avoided
- the straightened image represents the guide-defined spine corridor
- the straightened image is saved as a 16-bit TIFF using the same storage scale policy

Outputs are written to:

```text
straightened/
```

---

## 11. Process all - heatmap + tables

Use:

```text
Process all - heatmap + tables
```

SCAMP extracts longitudinal column-sum profiles from straightened TIFF files. In the current no-trim implementation, the full straightened image width is used for each profile; no relative-threshold trimming is applied during Process all.

The current implementation can use:

1. straightened images open in memory
2. straightened preview tabs
3. saved TIFF files in `straightened/`

This is important for lazy/closeable preview workflows.

Profile generation now uses the complete straightened output width. Profiles are therefore not shortened by an automatic signal-start threshold. If straightened images have different widths, shorter profiles are padded at their distal ends with empty/NaN entries in the combined output matrix so that all rows have the same number of columns.

Outputs include:

```text
COHORT_HEATMAP_<ExperimentID>.pdf
COHORT_HEATMAP_<ExperimentID>.png
COMBINED_PIXEL_DATA_<ExperimentID>.csv
COMBINED_PIXEL_DATA_<ExperimentID>.xlsx
```

---

## Main output folders

Inside each experiment folder:

```text
background_subtracted/
    <sample>_<ExperimentID>_background_subtracted_Zdepth_normalised_sumIP_<condition>.tif
    previews/

straightened/
    <sample>_<ExperimentID>_straightened_<condition>.tif
    previews/

qc_reports/
    <sample>_background_qc.json
    <sample>_background_qc.csv
    <sample>_background_advanced_qc.json
    <sample>_straighten_qc.json
    <sample>_straighten_qc.csv
    <sample>_straighten_advanced_qc.json
```

---

## Clean QC reports

SCAMP now separates user-facing QC from developer/debug QC.

### User-facing QC

The main `.json` and `.csv` QC files contain only the most important measurement-quality fields.

### Advanced QC

Full internal diagnostic data are still saved separately as:

```text
*_advanced_qc.json
```

Use these only for debugging or development.

---

## Background QC summary

The user-facing background QC focuses on:

```text
QC status: PASS / CHECK / FAIL
Background method
ROI area
Background level
Background stability
Signal retention
```

Recommended interpretation:

- `PASS`: no obvious technical issue
- `CHECK`: inspect manually
- `FAIL`: likely technical failure

Background QC is intentionally conservative. Sparse calcification images can naturally have many zero pixels after subtraction, so high zero fraction alone should not automatically fail a sample.

---

## Straightening QC summary

The user-facing straightening QC focuses on:

```text
QC status: PASS / CHECK / FAIL
Guide count
Corridor intensity change
Storage positive-pixel retention
Storage intensity drift
```

The most important straightening metric is:

```text
Corridor intensity change
```

This compares the guide-defined source spine corridor with the straightened output, not the whole input image.

Whole-image intensity change is not used for QC decisions, because the original image contains large areas outside the straightened spine corridor.

---

## QC details in the app

QC values are clickable in the app.

Clicking the QC status opens a simple details window. The window is informational only and can be closed normally.

No separate QC popup is shown automatically after each processing step.

---

## QC status philosophy

SCAMP uses conservative QC flags.

### PASS

No obvious technical issue.

### CHECK

Manual inspection recommended. This may reflect biological or image-dependent variation, not necessarily failure.

### FAIL

Reserved for technical failures, such as:

- missing ROI
- empty ROI
- missing Z-step for Z-depth normalisation
- empty output
- invalid values
- failed output writing
- severe storage roundtrip loss
- failed straightening

---

## Memory-safe usage recommendations

For large datasets:

1. Open all CZI files.
2. Load one preview.
3. Adjust transform and ROI.
4. Save ROI.
5. Close the preview.
6. Continue with the next file.
7. Run `Subtract background` after all ROIs are ready.

This keeps memory use much lower than keeping every CZI stack open.

---

## Notes

- Quantitative outputs are 16-bit TIFF, not PNG.
- Preview images are contrast-stretched and stored in `previews/`.
- Preview images are for viewing only and should not be used for measurement.
- Background subtraction is slice-wise.
- Z-depth normalisation requires valid Z-step metadata.
- Straightening is QC-checked against the guide-defined spine corridor.
- Process all exports full-width straightened column-sum profiles without trimming.
- Storage roundtrip QC verifies that TIFF saving did not destroy low-intensity float values.
