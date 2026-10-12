# SCAMP

Version: `20261002_132001`

SCAMP (Spinal Calcification & Mineralization Profiler) is a single-file Python GUI application for converting fluorescence z-stacks of Alizarin Red S-stained fish larvae into straightened spinal images and one-dimensional longitudinal intensity profiles.

## Main workflow

1. Create or select an experiment and define its experimental conditions.
2. Open a supported fluorescence z-stack, define or import a background ROI, or open an existing quantitative projection image.
3. Generate background-corrected projections.
4. Place landmark triplets along the notochord. The green point marks the approximate midline; the red and blue points define paired dorsal and ventral corridor boundaries.
5. Straighten the guide-defined spinal corridor.
6. Extract full-length column-sum intensity profiles and export cohort tables and heatmaps.

## Straightening algorithm

The quantitative straightening procedure uses the paired user-defined dorsal and ventral landmarks directly. Cubic-spline boundary curves shown in the GUI are visual guides and are not used as transform control points.

For each specimen, SCAMP:

- orders landmark triplets by their midline x-coordinate;
- calculates the midpoint of each dorsal-to-ventral pair;
- parameterizes the resulting centreline by cumulative Euclidean arc length;
- linearly interpolates paired dorsal and ventral coordinates along that arc length;
- creates a rectangular target grid whose width follows centreline arc length and whose height is based on the median corridor width;
- maps every target column to a dorsal-to-ventral source-image cross-section;
- samples source intensities using first-order interpolation.

No specimen-specific intensity normalization or thresholding is applied during straightening. The quantitative input scale is retained, but individual output pixels are interpolated values and geometric resampling does not guarantee preservation of the global pixel-intensity sum.

## Profile extraction

SCAMP sums pixel intensities down each column of the straightened image to produce a one-dimensional rostro-caudal fluorescence profile. Profiles retain their full specimen-specific lengths; no intensity-based trimming is applied. For matrix export, shorter profiles are padded distally with `NaN` values so padding remains distinct from true zero intensity.

Length normalization to 1000 relative positions is not performed by this Python application. If used, it is a separate downstream R-analysis step and removes absolute profile-length information from the normalized representation.

## Supported input formats

The complete quantitative z-stack workflow has direct support for:

- Zeiss CZI (`.czi`);
- multi-page TIFF (`.tif`, `.tiff`);
- OME-TIFF (`.ome.tif`, `.ome.tiff`).

ND2 (`.nd2`), LIF (`.lif`), and LSM (`.lsm`) files can be opened through the installed BioIO or AICSImageIO reader backend. These proprietary-format routes should be considered experimental until they have been tested end-to-end with representative files from the corresponding microscope software. Reader availability depends on the plugins and optional dependencies installed in the Python environment.

SCAMP is not tied to a particular microscope manufacturer or to structured-illumination imaging. Images acquired on other fluorescence microscope platforms can be analysed directly when supplied in a supported format, provided that quantitative pixel values and the metadata required for the selected workflow are preserved.

### Minimum input requirements

For quantitative z-stack processing, the input must:

- fully contain the spinal interval selected for analysis;
- retain quantitative pixel values without substantial saturation or clipping;
- provide sufficient spatial resolution to distinguish chordacentrum and intervertebral signals;
- contain a signal-free region suitable for background estimation;
- show the notochord boundaries sufficiently clearly for reproducible landmark placement;
- provide the number of optical sections and the z-step size for Z-depth normalization.

If the z-step is not present in the file metadata, SCAMP asks the user to enter it manually. In the study associated with SCAMP, fluorescence-derived endpoints were calculated from the interval encompassing the seven most anterior chordacentra; imaging of the entire vertebral column was therefore not required for these endpoints. Coverage of the complete counting interval is required when the total number of mineralized chordacentra is scored separately.

The current importer is intended primarily for single-channel fluorescence z-stacks. When a microscopy container contains multiple scenes, time points, or channels, users must verify that the imported data correspond to the intended specimen, time point, and Alizarin Red S channel before quantitative processing.

## Requirements

Python 3.11 is recommended.

Core packages:

- `numpy`
- `scipy`
- `matplotlib`
- `Pillow`
- `tifffile` for quantitative TIFF and OME-TIFF input/output
- Tkinter, normally supplied with Python

Packages used by particular input/output features:

- `pandas` and `openpyxl` for XLSX export
- `scikit-image` for ImageJ ROI polygon handling and image rotation utilities
- `roifile` for ImageJ/Fiji ROI input
- `bioio` or `aicsimageio` for microscopy-container files
- `aicspylibczi` for additional CZI support
- `opencv-python` for additional image I/O
- `ttkbootstrap` for the optional modern GUI theme

The application checks optional dependencies at startup. Missing optional packages disable only their associated features where possible. Support for proprietary formats also depends on the corresponding reader plugin being installed in the same Python environment from which SCAMP is launched.

## Running SCAMP

Activate the Python environment containing the dependencies, then run:

```bash
python SCAMP_20261002_132001.py
```

SCAMP requires a graphical desktop environment because its workflow uses Tkinter and embedded Matplotlib windows.

## Quantitative image handling

Background is estimated independently for each optical section from a user-defined ROI and subtracted from that section. Negative corrected values are clipped to zero. Projection and storage behavior depends on the selected projection mode. Generated quantitative TIFFs use a fixed storage scale when required to retain sub-integer values in 16-bit files; SCAMP converts these values back to native quantitative units before profile extraction.

The same background-correction, projection, straightening, QC, and profile-extraction operations are applied after import regardless of the source microscopy format. Within a quantitative experiment, specimens should be acquired using the same microscope, objective, illumination intensity, exposure time, detector settings, bit depth, and relevant stack settings.

Contrast-stretched previews are intended only for display and must not be used for quantitative analysis.

## Output organization

Outputs are grouped under the selected experiment identifier. Depending on the input and chosen processing steps, the experiment directory can include:

- background-corrected projection images;
- straightened quantitative TIFF images;
- per-step QC reports;
- combined CSV and XLSX profile matrices;
- cohort heatmaps in PDF or PNG format.

Generated filenames encode the source specimen, experiment identifier, processing stage, and assigned condition where applicable.

## Interpretation notes

- Straightening standardizes geometry but does not establish anatomical landmark correspondence between specimens.
- Raw Python profiles have specimen-specific arc-length-derived widths.
- Column sums depend on the sampled dorsal-to-ventral corridor, so landmark placement and corridor width must be applied consistently.
- A later fixed-length interpolation supports comparison at relative rostro-caudal positions but does not retain absolute spinal length.
- Quantitative analyses should use exported native-unit tables or quantitative TIFFs, never preview images.
