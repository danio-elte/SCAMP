#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCAMP — Spinal Calcification & Mineralization Profiler
======================================================

A single-file, napari-style GUI application for the Alizarin Red S zebrafish
spinal-mineralization pipeline, from raw CZI to a cohort heatmap.

(scamp.py)

It unifies the whole workflow that used to live in four separate scripts:

    1. norm_mean_IP.py              (CZI + ROI -> background-corrected 16-bit
                                     mean-intensity projection)
    2. interactive_straighten2.py   (manual landmark placement + straightening)
    3. gerinc_illesztes_batch_heatmap.py  (profile extraction + heatmap + tables)
    4. warp.py                      (piecewise-affine warp helper)

Workflow inside the app
-----------------------
    * On launch, establish an ExperimentID (proposed as "yyyymmdd_<random>",
      editable, guaranteed not to collide with an existing directory). A
      directory named after the ExperimentID is created; all outputs live
      inside it.
    * Set up the conditions for the experiment (defaults to "control",
      editable, add as many as needed).
    * Import raw CZI files (each with a matching ImageJ .roi for the
      background region), choosing the condition for each import batch. The
      app builds the normalized 16-bit projection and immediately saves it
      (the pure image, no overlay) to <ExperimentID>/normalized/.
      You can also Open existing 16-bit projection images directly.
    * For each image, drag the landmark triplets onto the notochord
      (green = midline, red = dorsal, blue = ventral). Add / remove
      triplets as needed.
    * Straightening applies a piecewise-affine transform (Delaunay
      triangulation + per-triangle affine + bilinear sampling) in full
      16-bit precision and immediately saves the pure straightened image to
      <ExperimentID>/straightened/, overwriting any previous version of that
      sample and replacing its preview tab.
    * Process all extracts full-length 1-D longitudinal intensity profiles
      (column sums), pads them to a common length when necessary, and writes
      into the experiment directory:
          - a cohort heatmap as PDF, sized to the number of samples and
            color-coded by condition when more than one is present
          - the combined matrix as CSV and XLSX (with a "group" column)

Design goal: be as fail-safe as possible. Almost every operation that can
fail (file I/O, image decoding, degenerate geometry, missing optional
dependencies) is guarded, and the user is told what happened rather than
seeing a stack trace and a dead window.

This script is meant to be launched like napari:

    conda activate spine        # or any env with the deps below
    python scamp.py

Core dependencies: numpy, scipy, scikit-image, matplotlib, Pillow.
Optional: pandas + openpyxl (XLSX export); opencv (extra image I/O);
aicsimageio/bioio + roifile (CZI import). Missing optional pieces only
disable the corresponding feature.
"""

import os
import re
import sys
import csv
import math
import random
import string
import traceback
import gc
import json
from datetime import date

# ----------------------------------------------------------------------
# Tkinter import (guarded -- a headless box may lack it)
# ----------------------------------------------------------------------
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as exc:  # pragma: no cover
    sys.stderr.write(
        "ERROR: This program needs a graphical environment with Tkinter.\n"
        f"Tkinter could not be imported: {exc}\n"
    )
    sys.exit(1)

# ----------------------------------------------------------------------
# Optional modern Tk theme layer
# ----------------------------------------------------------------------
try:
    import ttkbootstrap as ttkbs
    _HAVE_TTKBOOTSTRAP = True
except Exception:
    ttkbs = None
    _HAVE_TTKBOOTSTRAP = False


import numpy as np

# ----------------------------------------------------------------------
# Matplotlib (embedded in Tk)
# ----------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )
    import matplotlib.cm as cm
except Exception as exc:  # pragma: no cover
    sys.stderr.write(f"ERROR: matplotlib (with TkAgg) is required: {exc}\n")
    sys.exit(1)

# ----------------------------------------------------------------------
# SciPy / scikit-image (core numerical dependencies)
# ----------------------------------------------------------------------
try:
    from scipy.interpolate import CubicSpline
except Exception as exc:  # pragma: no cover
    sys.stderr.write(f"ERROR: scipy is required: {exc}\n")
    sys.exit(1)

try:
    from skimage.transform import PiecewiseAffineTransform, warp as sk_warp
except Exception as exc:  # pragma: no cover
    sys.stderr.write(f"ERROR: scikit-image is required: {exc}\n")
    sys.exit(1)

# ----------------------------------------------------------------------
# Optional dependencies
# ----------------------------------------------------------------------
try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False

try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False

try:
    import pandas as pd
    _HAVE_PANDAS = True
except Exception:
    _HAVE_PANDAS = False

try:
    import tifffile
    _HAVE_TIFFFILE = True
except Exception:
    _HAVE_TIFFFILE = False

# ----------------------------------------------------------------------
# Optional CZI-import dependencies.
#
# aicsimageio is in maintenance-only mode; its successor is bioio
# (+ bioio-czi). We try aicsimageio first (matches the original
# czi_roi_env.yaml), then fall back to bioio so the script keeps working
# as the ecosystem migrates. roifile reads the ImageJ .roi background
# region. Any of these missing simply disables the CZI-import button.
# ----------------------------------------------------------------------
_CZI_READER = None  # callable(path) -> reader with get_image_data
_CZI_READER_NAME = None       # which backend was loaded ("bioio" / "aicsimageio")
_READER_IMPORT_ERRORS = {}    # backend name -> import error string (for diagnostics)

# Prefer bioio (the actively maintained successor); fall back to aicsimageio.
try:
    from bioio import BioImage as _BioImage

    def _czi_open(path):
        return _BioImage(path)

    _CZI_READER = _czi_open
    _CZI_READER_NAME = "bioio"
except Exception as _exc:
    _READER_IMPORT_ERRORS["bioio"] = f"{type(_exc).__name__}: {_exc}"
    try:
        from aicsimageio import AICSImage as _AICSImage

        def _czi_open(path):
            return _AICSImage(path)

        _CZI_READER = _czi_open
        _CZI_READER_NAME = "aicsimageio"
    except Exception as _exc2:
        _READER_IMPORT_ERRORS["aicsimageio"] = f"{type(_exc2).__name__}: {_exc2}"
        _CZI_READER = None

try:
    from roifile import ImagejRoi
    _HAVE_ROIFILE = True
except Exception as _exc:
    _HAVE_ROIFILE = False
    _READER_IMPORT_ERRORS["roifile"] = f"{type(_exc).__name__}: {_exc}"

try:
    from skimage.draw import polygon as _sk_polygon
    _HAVE_SKDRAW = True
except Exception:
    _HAVE_SKDRAW = False

_HAVE_CZI = (_CZI_READER is not None) and _HAVE_ROIFILE and _HAVE_SKDRAW

try:
    from aicspylibczi import CziFile as _AICSPY_CZI_FILE
    _HAVE_AICSPYLIBCZI = True
except Exception as _exc:
    _AICSPY_CZI_FILE = None
    _HAVE_AICSPYLIBCZI = False
    _READER_IMPORT_ERRORS["aicspylibczi"] = f"{type(_exc).__name__}: {_exc}"


# ======================================================================
#  Image I/O helpers (fail-safe, format-agnostic)
# ======================================================================
def load_image_any(path):
    """
    Load an image as a 2-D numpy array, preserving bit depth where possible.

    Tries PIL first (good 16-bit PNG support), then OpenCV. Multi-channel
    images are converted to grayscale by averaging. Returns float64 plus the
    original dtype so we know the valid output range.

    Returns
    -------
    (img_float64, original_dtype)  or raises IOError with a clear message.
    """
    if not os.path.isfile(path):
        raise IOError(f"File does not exist: {path}")

    arr = None
    orig_dtype = None

    # Prefer tifffile for TIFFs (robust 16-bit grayscale reader).
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff") and _HAVE_TIFFFILE:
        try:
            arr = tifffile.imread(path)
            orig_dtype = arr.dtype
        except Exception:
            arr = None

    if arr is None and _HAVE_PIL:
        try:
            with Image.open(path) as im:
                im.load()
                arr = np.asarray(im)
                orig_dtype = arr.dtype
        except Exception:
            arr = None

    if arr is None and _HAVE_CV2:
        try:
            arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if arr is not None:
                orig_dtype = arr.dtype
        except Exception:
            arr = None

    if arr is None:
        raise IOError(
            f"Could not decode image (need Pillow or OpenCV): {path}"
        )

    arr = np.asarray(arr)

    # Reduce to 2-D grayscale.
    if arr.ndim == 3:
        # If there is an alpha channel, drop it.
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        arr = arr.mean(axis=2)
    elif arr.ndim != 2:
        raise IOError(f"Unsupported image shape {arr.shape}: {path}")

    return arr.astype(np.float64), orig_dtype


def dtype_max(orig_dtype):
    """Maximum representable value for the saved output, default 16-bit."""
    try:
        if orig_dtype is not None and np.issubdtype(orig_dtype, np.integer):
            return float(np.iinfo(orig_dtype).max)
    except Exception:
        pass
    return 65535.0


def _to_uint16_scaled(arr_float):
    """
    Map a float image to the full uint16 range based on its own min..max, for
    a viewable preview. Quantitative projections occupy only the bottom few
    percent of 0..65535, so a raw cast looks black in ordinary viewers; this
    stretch makes the preview visible. NOT comparable between samples.
    Returns (uint16_array, had_signal).
    """
    a = np.asarray(arr_float, dtype=np.float64)
    a = np.where(np.isfinite(a), a, 0.0)
    a[a < 0] = 0.0
    vmin = float(a.min())
    vmax = float(a.max())
    if vmax <= vmin:                      # flat / empty image
        return np.zeros(a.shape, np.uint16), False
    scaled = (a - vmin) / (vmax - vmin) * 65535.0
    return scaled.astype(np.uint16), True


def _to_uint16_raw(arr_float):
    """
    Faithful raw quantitative conversion: clip into 0..65535 and cast. Values
    are preserved (comparable between samples). Returns (uint16, vmin, vmax,
    had_signal).
    """
    a = np.asarray(arr_float, dtype=np.float64)
    a = np.where(np.isfinite(a), a, 0.0)
    out = np.clip(a, 0, 65535).astype(np.uint16)
    return out, float(a.min()), float(a.max()), bool(out.max() > 0)


def _write_uint16(path, out):
    """Write a uint16 2-D array to path, format by extension. TIFF preferred.
    Returns True on success."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff") and _HAVE_TIFFFILE:
        try:
            tifffile.imwrite(path, out)
            return True
        except Exception:
            pass
    if _HAVE_PIL:
        try:
            try:
                Image.fromarray(out).save(path)
            except Exception:
                Image.fromarray(out).convert("I;16").save(path)
            return True
        except Exception:
            pass
    if _HAVE_CV2:
        try:
            if cv2.imwrite(path, out):
                return True
        except Exception:
            pass
    return False


def preview_path_for(path, tag="_preview"):
    """Return the preview path inside a sibling previews/ folder.

    Example:
        a/b.tif -> a/previews/b_preview.tif

    Preview files are contrast-stretched convenience copies and are kept out
    of the quantitative output folders.
    """
    folder, filename = os.path.split(path)
    root, ext = os.path.splitext(filename)
    preview_dir = os.path.join(folder or ".", "previews")
    return os.path.join(preview_dir, root + tag + ext)


def _remove_image_and_preview(path):
    """Best-effort removal of a saved image and its _preview sibling."""
    for p in (path, preview_path_for(path)):
        try:
            if p and os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass


def save_image16(path, arr_float, with_preview=True):
    """
    Save a float image as a 16-bit grayscale file (TIFF recommended). The main
    file holds the RAW quantitative values (clipped into 0..65535), which are
    faithful and comparable between samples; ImageJ/FIJI auto-contrast it on
    open. When with_preview is True, an additional contrast-stretched copy is
    written alongside as "<name>_preview<ext>" for quick viewing in ordinary
    image viewers (this copy is NOT quantitatively comparable).

    Returns (had_signal, vmin, vmax). Raises IOError only if the main file
    cannot be written by any backend.
    """
    out, vmin, vmax, had_signal = _to_uint16_raw(arr_float)
    if not _write_uint16(path, out):
        raise IOError(f"Could not write image (need tifffile, Pillow, or "
                      f"OpenCV): {path}")

    if with_preview:
        prev, _ = _to_uint16_scaled(arr_float)
        try:
            preview_path = preview_path_for(path)
            os.makedirs(os.path.dirname(preview_path), exist_ok=True)
            _write_uint16(preview_path, prev)
        except Exception:
            pass  # preview is best-effort; the raw file is what matters

    return had_signal, vmin, vmax


def _sanitize(text):
    """Make a string safe to embed in a filename (no path separators etc.)."""
    text = str(text).strip()
    # collapse whitespace, drop characters that are awkward in filenames
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._-]", "", text)
    return text


def build_filename(base, experiment_id, condition, kind="", ext=".tif"):
    """
    Stitch the ExperimentID, an optional kind tag (e.g. "normalized" or
    "straightened"), and the condition onto a filename. The ExperimentID
    already begins with the date (yyyymmdd_...), so no separate date is added.

    Result:  <base>_<ExperimentID>[_<kind>][_<condition>]<ext>
    e.g.     fish03_20260620_a1b2c3_normalized_control.tif

    The kind and condition are each omitted if empty.
    """
    parts = [_sanitize(base), _sanitize(experiment_id)]
    k = _sanitize(kind)
    if k:
        parts.append(k)
    cond = _sanitize(condition)
    if cond:
        parts.append(cond)
    return "_".join(p for p in parts if p) + ext


def projection_extension_for_source(path):
    """Return the output extension for projections made from a source file.

    All generated projection outputs are written as quantitative 16-bit TIFFs,
    regardless of the source image/container format. This keeps downstream
    analysis consistent and avoids lossy/non-quantitative formats.
    """
    return ".tif"


def propose_experiment_id(parent_dir, when=None, n_rand=6):
    """
    Propose a unique ExperimentID of the form "yyyymmdd_<random>", where the
    random part is lowercase letters+digits. Guaranteed not to match any
    existing directory directly inside parent_dir (case-insensitively).
    """
    stamp = (when or date.today()).strftime("%Y%m%d")
    try:
        existing = {name.lower() for name in os.listdir(parent_dir)}
    except Exception:
        existing = set()
    alphabet = string.ascii_lowercase + string.digits
    for _ in range(10000):
        rand = "".join(random.choice(alphabet) for _ in range(n_rand))
        candidate = f"{stamp}_{rand}"
        if candidate.lower() not in existing:
            return candidate
    # extremely unlikely fallback: widen the random part
    return f"{stamp}_{''.join(random.choice(alphabet) for _ in range(n_rand + 4))}"


def experiment_id_is_free(parent_dir, experiment_id):
    """True if no directory named experiment_id exists in parent_dir."""
    target = _sanitize(experiment_id).lower()
    if not target:
        return False
    try:
        existing = {name.lower() for name in os.listdir(parent_dir)}
    except Exception:
        existing = set()
    return target not in existing


# ======================================================================
#  CZI -> normalized mean-intensity projection (port of norm_mean_IP.py)
# ======================================================================
def roi_to_mask(roi, shape):
    """Convert an ImageJ ROI to a filled boolean mask for a 2-D shape."""
    mask = np.zeros(shape, dtype=bool)
    coords = roi.coordinates()
    rr, cc = _sk_polygon(
        coords[:, 1].astype(int),
        coords[:, 0].astype(int),
        shape,
    )
    mask[rr, cc] = True
    return mask


def czi_to_projection(czi_path, roi_path):
    """
    Background-correct a z-stack CZI using a ROI background region and return
    a normalized mean-intensity projection as a float array (16-bit scale).

    For each optical section, the mean intensity inside the ROI is treated as
    background and subtracted; negatives are clipped to zero. The corrected
    stack is summed over z and divided by the number of slices.

    Raises IOError / ValueError with a clear message on failure.
    """
    if not _HAVE_CZI:
        raise RuntimeError(
            "CZI import needs aicsimageio (or bioio) + roifile + scikit-image."
        )
    if not os.path.isfile(czi_path):
        raise IOError(f"CZI not found: {czi_path}")
    if not os.path.isfile(roi_path):
        raise IOError(f"Matching .roi not found: {roi_path}")

    img = _CZI_READER(czi_path)
    # ZYX grayscale stack. get_image_data is provided by both readers.
    data = np.asarray(img.get_image_data("ZYX")).astype(np.float64)
    if data.ndim != 3 or data.shape[0] < 1:
        raise ValueError(f"Unexpected CZI shape {data.shape} in {czi_path}")
    Z = data.shape[0]

    roi = ImagejRoi.fromfile(roi_path)
    mask = roi_to_mask(roi, data.shape[1:])
    if not mask.any():
        raise ValueError("ROI mask is empty (check the .roi file).")

    bg_corrected = np.empty_like(data)
    for z in range(Z):
        sl = data[z]
        bg = float(np.mean(sl[mask]))
        corr = sl - bg
        corr[corr < 0] = 0
        bg_corrected[z] = corr

    mean_projection = np.sum(bg_corrected, axis=0) / float(Z)
    return mean_projection


def find_roi_for_czi(czi_path):
    """Return the matching <base>.roi path next to a CZI, or None."""
    base = os.path.splitext(czi_path)[0]
    cand = base + ".roi"
    return cand if os.path.isfile(cand) else None


def geometry_sidecar_path_for_czi(czi_path):
    """Return the SCAMP geometry sidecar path for a CZI file."""
    return os.path.splitext(czi_path)[0] + "_scamp_geometry.json"


def load_geometry_sidecar(czi_path):
    """Load saved transform history and ROI rectangle for a CZI, if present."""
    path = geometry_sidecar_path_for_czi(czi_path)
    if not os.path.isfile(path):
        return [], None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        history = [tuple(item) for item in data.get("transform_history", [])]
        rect = data.get("background_roi_rect")
        if rect is not None:
            rect = tuple(float(v) for v in rect)
        return history, rect
    except Exception:
        return [], None


def save_geometry_sidecar(czi_path, transform_history, background_roi_rect):
    """Persist SCAMP geometry state next to a CZI file."""
    path = geometry_sidecar_path_for_czi(czi_path)
    data = {
        "source_czi": os.path.basename(czi_path),
        "transform_history": [list(op) for op in (transform_history or [])],
        "background_roi_rect": (list(background_roi_rect)
                                  if background_roi_rect is not None else None),
        "note": (
            "SCAMP geometry sidecar. Background subtraction should replay "
            "transform_history before applying background_roi_rect."),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return path


# ======================================================================
#  Geometry: triplets, splines, arc-length target, piecewise-affine warp
# ======================================================================
class Triplet:
    """
    A landmark triplet: a midline point with dorsal ('above') and ventral
    ('below') points kept at a fixed half-length along a shared direction.
    """

    __slots__ = ("middle", "above", "below", "half_length")

    def __init__(self, x, y, image_height):
        init_half = max(image_height / 7.0, 5.0)
        self.middle = np.array([float(x), float(y)], dtype=float)
        self.half_length = float(init_half)
        # default direction points "up" (smaller y)
        direction = np.array([0.0, -1.0])
        self.above = self.middle + direction * self.half_length
        self.below = self.middle - direction * self.half_length

    # -- moving the whole triplet by its middle -----------------------
    def move_middle(self, pos):
        direction = self.above - self.middle
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.array([0.0, -1.0])
        else:
            direction = direction / norm
        self.middle = np.asarray(pos, dtype=float)
        self.above = self.middle + direction * self.half_length
        self.below = self.middle - direction * self.half_length

    # -- dragging the dorsal point ------------------------------------
    def move_above(self, pos):
        """Rotate the triplet around the middle point without changing size."""
        candidate = np.asarray(pos, dtype=float)
        direction = candidate - self.middle
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return
        direction = direction / norm
        # The reference-point distance is fixed: dragging changes only angle.
        self.above = self.middle + direction * self.half_length
        self.below = self.middle - direction * self.half_length

    # -- dragging the ventral point -----------------------------------
    def move_below(self, pos):
        """Rotate the triplet around the middle point without changing size."""
        candidate = np.asarray(pos, dtype=float)
        direction = candidate - self.middle
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return
        direction = direction / norm
        # Keep the dragged ventral point at the fixed distance from center.
        self.below = self.middle + direction * self.half_length
        self.above = self.middle - direction * self.half_length


def interpolate_boundaries(triplets, num_interpolated=7):
    """
    Cubic-spline interpolate the dorsal ('above') and ventral ('below')
    boundary point sets along x. Returns (above_pts, below_pts) as arrays,
    or (None, None) if fewer than 2 triplets.
    """
    if len(triplets) < 2:
        return None, None

    xA = np.array([t.above[0] for t in triplets], dtype=float)
    yA = np.array([t.above[1] for t in triplets], dtype=float)
    xB = np.array([t.below[0] for t in triplets], dtype=float)
    yB = np.array([t.below[1] for t in triplets], dtype=float)

    # CubicSpline requires strictly increasing x. Sort and de-duplicate.
    def _clean(x, y):
        order = np.argsort(x)
        x, y = x[order], y[order]
        keep = np.concatenate(([True], np.diff(x) > 1e-9))
        return x[keep], y[keep]

    xA, yA = _clean(xA, yA)
    xB, yB = _clean(xB, yB)

    if len(xA) < 2 or len(xB) < 2:
        return None, None

    try:
        csA = CubicSpline(xA, yA)
        csB = CubicSpline(xB, yB)
    except Exception:
        return None, None

    above, below = [], []
    for i in range(1, len(xA)):
        xa = np.linspace(xA[i - 1], xA[i], num=num_interpolated, endpoint=False)
        above.extend(zip(xa, csA(xa)))
    for i in range(1, len(xB)):
        xb = np.linspace(xB[i - 1], xB[i], num=num_interpolated, endpoint=False)
        below.extend(zip(xb, csB(xb)))

    return np.array(above), np.array(below)


def arc_length_targets(upper, lower):
    """
    Build straight target coordinates whose x positions follow the cumulative
    arc length of the dorsal boundary, preserving longitudinal distances.
    """
    seg = np.linalg.norm(upper[1:] - upper[:-1], axis=1)
    s = np.insert(np.cumsum(seg), 0, 0.0)
    height = float(np.mean(lower[:, 1] - upper[:, 1]))
    if not np.isfinite(height) or abs(height) < 1.0:
        height = 1.0
    tgt_up = np.column_stack([s, np.zeros_like(s)])
    tgt_lo = np.column_stack([s, np.full_like(s, height)])
    return tgt_up, tgt_lo


def piecewise_affine_estimate(dst_pts, src_pts):
    """
    Estimate a piecewise-affine transform mapping dst -> src (the inverse
    map skimage.warp needs). Works across skimage versions: prefers the new
    `from_estimate` constructor, falls back to the deprecated `.estimate`.
    Returns a transform object or None on failure.
    """
    # New API (skimage >= 0.26)
    if hasattr(PiecewiseAffineTransform, "from_estimate"):
        try:
            tform = PiecewiseAffineTransform.from_estimate(dst_pts, src_pts)
            # from_estimate may return a FailedEstimation-like object that is
            # falsy; guard with bool().
            if tform:
                return tform
        except Exception:
            pass
    # Legacy API
    try:
        tform = PiecewiseAffineTransform()
        ok = tform.estimate(dst_pts, src_pts)
        if ok:
            return tform
    except Exception:
        pass
    return None


def straighten_image(img_float, upper, lower, max_dim=6000):
    """
    Straighten one image using dense ribbon resampling between paired dorsal
    and ventral guide boundaries.

    This intentionally avoids Delaunay / piecewise-affine triangulation. The
    previous mesh-based approach could leave unmapped pixels inside the target
    strip when triangles folded or did not cover the corridor cleanly. Dense
    ribbon sampling maps every output pixel to a source coordinate on the line
    between the paired upper/lower boundaries, so the full background inside
    the guide-defined corridor is preserved instead of creating black holes.

    The intensity range is preserved. Pixel values are bilinearly sampled, so
    local values can be interpolated, but no per-image normalization or
    thresholding is applied.
    """
    from scipy.ndimage import map_coordinates

    img = np.asarray(img_float, dtype=np.float64)
    upper = np.asarray(upper, dtype=float)
    lower = np.asarray(lower, dtype=float)

    if upper.shape != lower.shape or upper.ndim != 2 or upper.shape[1] != 2:
        raise ValueError("Upper/lower guide arrays must have matching Nx2 shape.")
    if upper.shape[0] < 2:
        raise ValueError("Need at least 2 paired guide triplets to straighten.")

    # Pairwise centreline and width. Use centreline arc length for the output
    # x-axis; this keeps dorsal/ventral cross-sections paired and avoids the
    # independent-spline pairing issue that can create twisted mesh geometry.
    center = 0.5 * (upper + lower)
    widths = np.linalg.norm(lower - upper, axis=1)
    if not np.all(np.isfinite(widths)) or np.nanmedian(widths) < 2:
        raise ValueError("Guide corridor width is too small or invalid.")

    seg = np.linalg.norm(center[1:] - center[:-1], axis=1)
    keep = np.concatenate(([True], seg > 1e-6))
    center = center[keep]
    upper = upper[keep]
    lower = lower[keep]
    if center.shape[0] < 2:
        raise ValueError("Guide points are degenerate after de-duplication.")

    s = np.insert(np.cumsum(np.linalg.norm(center[1:] - center[:-1], axis=1)), 0, 0.0)
    total_len = float(s[-1])
    out_w = int(math.ceil(total_len)) + 1
    out_h = int(math.ceil(float(np.nanmedian(np.linalg.norm(lower - upper, axis=1))))) + 1

    if out_w < 2 or out_h < 2:
        raise ValueError("Degenerate target geometry (zero size).")
    if out_w > max_dim or out_h > max_dim:
        raise ValueError(
            f"Target image too large ({out_w}x{out_h}); check landmark spread."
        )

    # Interpolate paired dorsal/ventral boundary coordinates as functions of
    # centreline arc length. This is not the same as using independent x-based
    # spline samples as control points for a triangulation; every output column
    # remains a direct cross-section between a paired upper/lower coordinate.
    xs = np.linspace(0.0, total_len, out_w)
    up_x = np.interp(xs, s, upper[:, 0])
    up_y = np.interp(xs, s, upper[:, 1])
    lo_x = np.interp(xs, s, lower[:, 0])
    lo_y = np.interp(xs, s, lower[:, 1])

    v = np.linspace(0.0, 1.0, out_h)[:, np.newaxis]
    sample_x = (1.0 - v) * up_x[np.newaxis, :] + v * lo_x[np.newaxis, :]
    sample_y = (1.0 - v) * up_y[np.newaxis, :] + v * lo_y[np.newaxis, :]

    out = map_coordinates(
        img,
        [sample_y, sample_x],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return np.asarray(out, dtype=np.float64)

# ======================================================================
#  Profile extraction / matrix assembly
# ======================================================================
def column_sums(img_float):
    """Integrate intensity perpendicular to the long axis (sum each column)."""
    return np.sum(np.asarray(img_float, dtype=np.float64), axis=0)


def trim_relative_threshold(signal, relative_ratio=0.05, consecutive=5):
    """
    Find the profile origin: first index where `consecutive` pixels in a row
    all exceed relative_ratio * max(signal). Returns (trimmed_signal, start)
    or (empty, None) if no such run exists.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n == 0:
        return signal, None

    smax = np.max(signal)
    if not np.isfinite(smax) or smax <= 0:
        return np.array([]), None

    threshold = relative_ratio * smax
    above = signal > threshold

    if consecutive <= 1:
        idx = np.argmax(above)
        if above[idx]:
            return signal[idx:], int(idx)
        return np.array([]), None

    # Sliding window of all-True runs.
    last_start = n - consecutive
    if last_start < 0:
        return np.array([]), None
    for i in range(last_start + 1):
        if np.all(above[i:i + consecutive]):
            return signal[i:], i
    return np.array([]), None


def assemble_matrix(profiles):
    """
    Pad a list of (name, group, 1-D array) profiles to common length with NaN.
    Returns (names, groups, matrix [n_samples x max_len]).
    """
    if not profiles:
        return [], [], np.zeros((0, 0))
    max_len = max((len(p) for _, _, p in profiles), default=0)
    names, groups, rows = [], [], []
    for name, group, p in profiles:
        p = np.asarray(p, dtype=np.float64)
        if len(p) < max_len:
            p = np.pad(p, (0, max_len - len(p)), constant_values=np.nan)
        names.append(name)
        groups.append(group if group else "")
        rows.append(p)
    return names, groups, (np.vstack(rows) if rows else np.zeros((0, 0)))


def write_tables(out_dir, experiment_id, names, groups, matrix):
    """
    Write the combined matrix to CSV (always) and XLSX (if pandas/openpyxl
    available). Files are named <COMBINED_PIXEL_DATA>_<ExperimentID>.{csv,xlsx}.
    Columns: image_name, group, p.1 .. p.N. Returns written paths.
    """
    written = []
    max_len = matrix.shape[1] if matrix.ndim == 2 else 0
    header = ["image_name", "group"] + [f"p.{i + 1}" for i in range(max_len)]

    csv_name = build_filename("COMBINED_PIXEL_DATA", experiment_id, "",
                              ext=".csv")
    csv_path = os.path.join(out_dir, csv_name)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for name, group, row in zip(names, groups, matrix):
            w.writerow([name, group] +
                       ["" if np.isnan(v) else v for v in row])
    written.append(csv_path)

    if _HAVE_PANDAS:
        try:
            df = pd.DataFrame(matrix, columns=header[2:])
            df.insert(0, "group", groups)
            df.insert(0, "image_name", names)
            xlsx_name = build_filename("COMBINED_PIXEL_DATA", experiment_id, "",
                                       ext=".xlsx")
            xlsx_path = os.path.join(out_dir, xlsx_name)
            df.to_excel(xlsx_path, index=False)
            written.append(xlsx_path)
        except Exception:
            # openpyxl missing or other issue -- CSV is still there.
            pass
    return written


# ======================================================================
#  GUI: per-image landmark editor (one tab in a notebook)
# ======================================================================
class SampleEditor(ttk.Frame):
    """
    One loaded sample: an embedded matplotlib canvas showing the image with
    draggable landmark triplets. Holds its own straightening result.
    """

    HIT_RADIUS_FRAC = 0.012  # fraction of image diagonal for hit-testing

    def __init__(self, master, path, app, generated_preview=False,
                 image_array=None, name=None, group="", auto_guides=False):
        super().__init__(master)
        self.app = app
        self.path = path
        self.name = name if name else os.path.basename(path)
        self.generated_preview = bool(generated_preview)
        self.has_straighten_guides = bool(auto_guides)
        self.straightened = None      # float array once computed
        self.normalized = None        # CZI projection array, if from CZI
        self.source_base = os.path.splitext(self.name)[0]  # filename stem
        self.group = group or ""      # condition / group label
        self.max_val = 65535.0
        # straighten bookkeeping (used on source editors, not previews)
        self.preview_editor = None    # the SampleEditor showing the result
        self.straightened_path = None # last saved straightened file path

        # ---- load image -------------------------------------------------
        if image_array is not None:
            self.img = np.asarray(image_array, dtype=np.float64)
            self.max_val = 65535.0
        else:
            self.img, orig_dtype = load_image_any(path)
            self.max_val = dtype_max(orig_dtype)
        if self.img.ndim != 2:
            raise IOError(f"Expected a 2-D image, got shape {self.img.shape}")
        self.h, self.w = self.img.shape

        # display normalization (robust percentile stretch)
        lo, hi = np.percentile(self.img, [1, 99.5])
        if hi <= lo:
            lo, hi = float(self.img.min()), float(self.img.max() + 1e-6)
        self.disp = np.clip((self.img - lo) / (hi - lo), 0, 1)

        self.hit_radius = self.HIT_RADIUS_FRAC * math.hypot(self.h, self.w)

        # ---- straighten guides ------------------------------------------
        # Projection images open without guides. The user adds them explicitly
        # with the main-window "Add straighten guides" command.
        self.triplets = []
        if self.has_straighten_guides:
            self.initialize_straighten_guides()

        # drag state
        self._drag = None  # (triplet, which) where which in {mid,above,below}

        self._build_ui()
        self._connect_events()
        self.redraw()

    # ------------------------------------------------------------------
    def _build_ui(self):
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar_frame, text="Add triplet",
                   command=self.add_triplet).pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Button(toolbar_frame, text="Remove last",
                   command=self.remove_triplet).pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Button(toolbar_frame, text="Reset triplets",
                   command=self.reset_triplets).pack(side=tk.LEFT, padx=2, pady=2)
        if self.generated_preview:
            initial_status = "Straightened preview file."
        elif self.has_straighten_guides:
            initial_status = "Drag points onto the notochord."
        else:
            initial_status = "Projection loaded. Use Add straighten guides before straightening."
        self.status_var = tk.StringVar(value=initial_status)
        ttk.Label(toolbar_frame, textvariable=self.status_var).pack(
            side=tk.RIGHT, padx=6)

        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(self.name, fontsize=9)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        nav = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        nav.update()
        nav.pack(side=tk.BOTTOM, fill=tk.X)

    def _connect_events(self):
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

    # ------------------------------------------------------------------
    def initialize_straighten_guides(self):
        """Create the default landmark guide triplets on demand."""
        self.triplets = []
        for x in np.linspace(0, self.w - 1, 8):
            self.triplets.append(Triplet(x, self.h / 2.0, self.h))
        self.has_straighten_guides = True
        self.straightened = None
        if hasattr(self, "status_var"):
            self.status_var.set("Drag points onto the notochord.")
        if hasattr(self, "canvas"):
            self.redraw()

    def add_triplet(self):
        self.has_straighten_guides = True
        x = self.w / 2.0
        self.triplets.append(Triplet(x, self.h / 2.0, self.h))
        self.triplets.sort(key=lambda t: t.middle[0])
        self.redraw()

    def remove_triplet(self):
        if self.triplets:
            self.triplets.pop()
            self.redraw()

    def reset_triplets(self):
        self.has_straighten_guides = True
        self.triplets = []
        for x in np.linspace(0, self.w - 1, 8):
            self.triplets.append(Triplet(x, self.h / 2.0, self.h))
        self.straightened = None
        self.redraw()

    # ------------------------------------------------------------------
    def _nearest_handle(self, x, y):
        """Return (triplet, which) for the closest handle within hit radius."""
        best = None
        best_d = self.hit_radius
        p = np.array([x, y])
        for t in self.triplets:
            for which, pt_ in (("mid", t.middle), ("above", t.above),
                               ("below", t.below)):
                d = np.linalg.norm(pt_ - p)
                if d < best_d:
                    best_d = d
                    best = (t, which)
        return best

    def on_press(self, event):
        if not self.has_straighten_guides:
            return
        if event.inaxes != self.ax or event.xdata is None:
            return
        # Ignore clicks while a navigation tool (pan/zoom) is active.
        if self.fig.canvas.toolbar is not None and \
                getattr(self.fig.canvas.toolbar, "mode", ""):
            return
        if event.button != 1:
            return
        self._drag = self._nearest_handle(event.xdata, event.ydata)

    def on_motion(self, event):
        if self._drag is None or event.inaxes != self.ax or event.xdata is None:
            return
        t, which = self._drag
        pos = np.array([event.xdata, event.ydata], dtype=float)
        if which == "mid":
            t.move_middle(pos)
        elif which == "above":
            t.move_above(pos)
        elif which == "below":
            t.move_below(pos)
        self.redraw()

    def on_release(self, event):
        if self._drag is not None:
            # keep triplets ordered by x so splines stay monotone
            self.triplets.sort(key=lambda t: t.middle[0])
        self._drag = None

    # ------------------------------------------------------------------
    def redraw(self):
        # Preserve the current Matplotlib view so editing guide points does not
        # reset an active zoom/pan state. This is especially important when the
        # user zooms into a small region before adjusting straighten guides.
        preserve_view = bool(self.ax.images)
        old_xlim = self.ax.get_xlim() if preserve_view else None
        old_ylim = self.ax.get_ylim() if preserve_view else None

        self.ax.clear()
        self.ax.imshow(self.disp, cmap="gray", origin="upper")
        self.ax.set_title(self.name, fontsize=9)
        if preserve_view:
            self.ax.set_xlim(old_xlim)
            self.ax.set_ylim(old_ylim)
        else:
            self.ax.set_xlim(0, self.w)
            self.ax.set_ylim(self.h, 0)

        if self.has_straighten_guides:
            # interpolated boundary preview
            above, below = interpolate_boundaries(self.triplets, num_interpolated=7)
            if above is not None and len(above):
                self.ax.plot(above[:, 0], above[:, 1], ".", ms=2, color="red")
            if below is not None and len(below):
                self.ax.plot(below[:, 0], below[:, 1], ".", ms=2, color="blue")

            for t in self.triplets:
                self.ax.plot([t.above[0], t.below[0]],
                             [t.above[1], t.below[1]],
                             "-", color="yellow", lw=1)
                self.ax.plot(*t.above, "o", color="red", ms=6)
                self.ax.plot(*t.below, "o", color="blue", ms=6)
                self.ax.plot(*t.middle, "o", color="lime", ms=6)

        self.ax.set_axis_off()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    def compute_straighten(self):
        """Run straightening; returns True on success, else sets status.

        The actual straightening uses the user-controlled guide triplet
        endpoints as paired dorsal/ventral cross-sections. Spline-interpolated
        points are only used for the visual guide preview.
        """
        if not self.has_straighten_guides or not self.triplets:
            self.status_var.set("Add straighten guides before straightening.")
            return False
        if len(self.triplets) < 2:
            self.status_var.set("Need >= 2 triplets to straighten.")
            return False
        ordered = sorted(self.triplets, key=lambda t: t.middle[0])
        above = np.array([t.above for t in ordered], dtype=float)
        below = np.array([t.below for t in ordered], dtype=float)
        try:
            self.straightened = straighten_image(self.img, above, below)
            self.status_var.set("Straightened OK.")
            return True
        except Exception as exc:
            self.straightened = None
            self.status_var.set(f"Straighten failed: {exc}")
            return False




# ======================================================================
#  Stack / hyperstack I/O and normalization-first preview tools
# ======================================================================
def _as_zyx_stack(arr):
    """
    Convert a loaded microscopy array to a grayscale ZYX stack.

    This helper intentionally accepts many common microscopy layouts. It keeps
    the last two axes as YX, treats one remaining axis as Z, and collapses any
    extra axes (time, channel, scene-like singleton dimensions) by maximum
    intensity. This makes Open tolerant of hyperstacks while keeping the first
    implementation predictable.
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        return a[np.newaxis, :, :].astype(np.float64)
    if a.ndim == 3:
        # Prefer ZYX. If the last axis looks like RGB/RGBA, convert to gray.
        if a.shape[-1] in (3, 4) and a.shape[0] > 4:
            rgb = a[..., :3]
            return rgb.mean(axis=-1)[np.newaxis, :, :].astype(np.float64)
        return a.astype(np.float64)

    # Remove singleton axes first.
    a = np.squeeze(a)
    if a.ndim == 2:
        return a[np.newaxis, :, :].astype(np.float64)
    if a.ndim == 3:
        return _as_zyx_stack(a)

    # For larger hyperstacks, keep YX as the last two axes and collapse all
    # axes except the most plausible Z axis. The most plausible Z axis is the
    # last non-YX axis with length > 1.
    spatial = a.shape[-2:]
    lead_shape = a.shape[:-2]
    z_axis = None
    for i in reversed(range(len(lead_shape))):
        if lead_shape[i] > 1:
            z_axis = i
            break
    if z_axis is None:
        return a.reshape((-1,) + spatial).mean(axis=0)[np.newaxis, :, :].astype(np.float64)

    # Move the chosen Z axis to the front, flatten the remaining lead axes, and
    # combine them by max intensity. This is a safe preview default for channels
    # and time points until explicit C/T controls are added.
    a = np.moveaxis(a, z_axis, 0)
    z = a.shape[0]
    rest = int(np.prod(a.shape[1:-2]))
    a = a.reshape((z, rest) + spatial)
    if rest > 1:
        a = np.max(a, axis=1)
    else:
        a = a[:, 0]
    return a.astype(np.float64)


def get_z_step_um_from_reader(reader):
    """Return the physical Z step in micrometers when the reader exposes it."""
    try:
        z_step = reader.physical_pixel_sizes.Z
        if z_step is not None and z_step > 0:
            return float(z_step)
    except Exception:
        pass
    return None


def load_microscopy_stack(path):
    """
    Load a microscopy image or hyperstack as (ZYX float stack, original dtype,
    z_step_um). Supports TIFF/OME-TIFF via tifffile and microscopy container
    formats through aicsimageio/bioio when installed. CZI files are never sent
    to the ordinary PIL/OpenCV loader, because that only produces a misleading
    "Could not decode image" message.
    """
    if not os.path.isfile(path):
        raise IOError(f"File does not exist: {path}")
    ext = os.path.splitext(path)[1].lower()
    arr = None
    orig_dtype = None
    z_step_um = None
    reader_errors = []

    microscopy_exts = {".czi", ".lif", ".nd2", ".lsm", ".ome"}
    lower = path.lower()
    use_reader = ext in microscopy_exts or lower.endswith((".ome.tif", ".ome.tiff"))

    if use_reader and _CZI_READER is not None:
        try:
            reader = _CZI_READER(path)
            z_step_um = get_z_step_um_from_reader(reader)
            read_attempts = [
                lambda: reader.get_image_data("ZYX"),
                lambda: reader.get_image_data("ZYX", T=0, C=0, S=0),
                lambda: reader.get_image_data("CZYX"),
                lambda: reader.get_image_data("TCZYX"),
                lambda: reader.data,
            ]
            last_exc = None
            for attempt in read_attempts:
                try:
                    arr = np.asarray(attempt())
                    orig_dtype = arr.dtype
                    break
                except TypeError as exc:
                    # Some readers do not accept S/T/C keyword arguments.
                    last_exc = exc
                except Exception as exc:
                    last_exc = exc
            if arr is None and last_exc is not None:
                raise last_exc
        except Exception as exc:
            reader_errors.append(f"aicsimageio/bioio failed: {exc}")
            arr = None

    # Direct CZI fallback. This is useful when aicsimageio is present but its
    # plugin dispatch fails, while aicspylibczi itself is installed.
    if arr is None and ext == ".czi" and _HAVE_AICSPYLIBCZI:
        try:
            czi = _AICSPY_CZI_FILE(path)
            result = czi.read_image()
            arr = result[0] if isinstance(result, tuple) else result
            arr = np.asarray(arr)
            orig_dtype = arr.dtype
        except Exception as exc:
            reader_errors.append(f"aicspylibczi failed: {exc}")
            arr = None

    if arr is None and _HAVE_TIFFFILE and ext in (".tif", ".tiff"):
        try:
            arr = tifffile.imread(path)
            orig_dtype = arr.dtype
        except Exception as exc:
            reader_errors.append(f"tifffile failed: {exc}")
            arr = None

    if arr is None:
        if use_reader:
            if reader_errors:
                details = "; ".join(reader_errors)
            else:
                # No reader branch even ran -> nothing is importable. Surface
                # the real import failures and which Python is running, since
                # the usual cause is a wrong/!activated environment.
                parts = []
                for name, err in _READER_IMPORT_ERRORS.items():
                    parts.append(f"{name}: {err}")
                why = (" | ".join(parts)
                       if parts else "no reader package importable")
                details = (f"no microscopy reader is installed for this "
                           f"format in the running interpreter "
                           f"({sys.executable}). Import attempts -> {why}")
            raise IOError(
                "Could not open microscopy stack. Install/repair the matching "
                "reader package (recommended: bioio + bioio-czi; alternative: "
                "aicsimageio + aicspylibczi) IN THE SAME ENVIRONMENT you launch "
                f"SCAMP from. Details: {details}"
            )
        # Fall back to the existing 2-D loader only for ordinary image files.
        img, orig_dtype = load_image_any(path)
        arr = img

    stack = _as_zyx_stack(arr)
    if stack.ndim != 3 or stack.shape[1] < 1 or stack.shape[2] < 1:
        raise IOError(f"Could not interpret image as a ZYX stack: {path}")
    return stack, orig_dtype, z_step_um


def normalize_display_image(img):
    """Robust 0..1 display stretch for a 2-D image."""
    a = np.asarray(img, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, dtype=float)
    lo, hi = np.percentile(finite, [1, 99.5])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max() + 1e-6)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def rectangle_to_slices(rect, shape):
    """Convert an (x0, y0, x1, y1) rectangle to clipped y/x slices."""
    if rect is None:
        return None
    x0, y0, x1, y1 = rect
    h, w = shape
    xa, xb = sorted([int(round(x0)), int(round(x1))])
    ya, yb = sorted([int(round(y0)), int(round(y1))])
    xa, xb = max(0, xa), min(w, xb)
    ya, yb = max(0, ya), min(h, yb)
    if xb <= xa or yb <= ya:
        return None
    return slice(ya, yb), slice(xa, xb)


def project_stack(stack, method="mean", background_rect=None, z_step_um=None):
    """
    Make a 2-D projection from a ZYX stack.

    If a rectangle is present, it is treated as the background region and its
    per-slice mean is subtracted before projection, matching the normalization
    script's background-correction logic. Negative corrected values are clipped
    to zero.
    """
    data = np.asarray(stack, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError("Expected a ZYX stack.")
    work = data.copy()
    slices = rectangle_to_slices(background_rect, data.shape[1:])
    if slices is not None:
        ys, xs = slices
        roi = work[:, ys, xs]
        if roi.size > 0:
            bg = np.mean(roi, axis=(1, 2))
            work = work - bg[:, np.newaxis, np.newaxis]
            work[work < 0] = 0

    method = method.lower()
    if method == "mean":
        return np.mean(work, axis=0)
    if method == "sum":
        return np.sum(work, axis=0)
    if method in ("zdepth", "z-depth", "zdepth normalised sum"):
        if z_step_um is None or z_step_um <= 0:
            raise ValueError("Z-depth normalised sum needs a positive Z-step in µm.")
        depth_um = float(work.shape[0]) * float(z_step_um)
        return np.sum(work, axis=0) / depth_um
    raise ValueError(f"Unknown projection method: {method}")


def project_stack_with_background_mask(stack, mask, method="zdepth", z_step_um=None):
    """
    Make a background-corrected projection from a ZYX stack using a 2-D mask.

    The mean intensity inside the ROI mask is subtracted independently from
    each Z slice. Negative corrected values are clipped to zero. For the
    Z-depth normalised sum, the corrected sum projection is divided by the
    physical stack depth in micrometers.
    """
    data = np.asarray(stack, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError("Expected a ZYX stack.")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != data.shape[1:]:
        raise ValueError(
            f"ROI mask shape {mask.shape} does not match image shape {data.shape[1:]}.")
    if not mask.any():
        raise ValueError("ROI mask is empty.")

    work = data.copy()
    roi = work[:, mask]
    bg = np.mean(roi, axis=1)
    work = work - bg[:, np.newaxis, np.newaxis]
    work[work < 0] = 0

    method = method.lower()
    if method == "mean":
        return np.mean(work, axis=0)
    if method == "sum":
        return np.sum(work, axis=0)
    if method in ("zdepth", "z-depth", "zdepth normalised sum"):
        if z_step_um is None or z_step_um <= 0:
            raise ValueError("Z-depth normalised sum needs a positive Z-step in µm.")
        depth_um = float(work.shape[0]) * float(z_step_um)
        return np.sum(work, axis=0) / depth_um
    raise ValueError(f"Unknown projection method: {method}")


def load_roi_mask_from_file(roi_path, shape):
    """Read an ImageJ ROI file and convert it to a filled boolean mask."""
    if not _HAVE_ROIFILE:
        raise RuntimeError("Reading ImageJ ROI files requires the roifile package.")
    if not _HAVE_SKDRAW:
        raise RuntimeError("ROI masks require scikit-image draw support.")
    if not os.path.isfile(roi_path):
        raise IOError(f"ROI file does not exist: {roi_path}")
    roi = ImagejRoi.fromfile(roi_path)
    mask = roi_to_mask(roi, shape)
    if not mask.any():
        raise ValueError(f"ROI mask is empty: {roi_path}")
    return mask



class DeferredCziEditor(ttk.Frame):
    """
    Lightweight placeholder for a CZI file.

    Opening many CZI files should not immediately load every full Z stack into
    RAM. This editor registers the file, shows any matching ROI in the file
    list, and loads the real StackEditor only when the user explicitly asks for
    a preview/ROI editing session.
    """

    def __init__(self, master, path, app, name=None, group=""):
        super().__init__(master)
        self.app = app
        self.path = path
        self.name = name if name else os.path.basename(path)
        self.group = group or ""
        self.source_base = os.path.splitext(self.name)[0]
        self.generated_preview = False
        self.is_stack_editor = True
        self.is_deferred_czi = True
        self.geometry_matches_source = True
        self.stack_loaded = False
        self.z_step_um = None
        self.preview_editor = None  # open StackEditor preview tab, if loaded
        self.saved_roi_path = None
        # Geometry state is empty for lazy CZI placeholders. If the user loads
        # a preview and applies transforms, the real StackEditor records the
        # operations and batch background subtraction replays them.
        self.transform_history = []
        self.background_roi_rect = None

        candidate_roi = os.path.splitext(self.path)[0] + ".roi"
        if os.path.isfile(candidate_roi):
            self.saved_roi_path = candidate_roi
        history, rect = load_geometry_sidecar(self.path)
        if history:
            self.transform_history = history
        if rect is not None:
            self.background_roi_rect = rect

        self._build_ui()

    def _build_ui(self):
        frame = ttk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)

        ttk.Label(
            frame,
            text=self.name,
            font=("TkDefaultFont", 12, "bold"),
        ).pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(
            frame,
            text=(
                "This CZI is registered but not loaded into memory.\n"
                "This keeps SCAMP responsive when many large CZI files are opened.\n\n"
                "Use 'Load preview / edit ROI' only for the file you want to inspect "
                "or annotate. Batch background subtraction can process this file "
                "from disk when a matching .roi file exists."
            ),
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 12))

        ttk.Button(
            frame,
            text="Load preview / edit ROI",
            command=self.load_full_stack_editor,
        ).pack(anchor=tk.W)

        self.status_var = tk.StringVar(value=self._status_text())
        ttk.Label(
            frame,
            textvariable=self.status_var,
            foreground="#555",
        ).pack(anchor=tk.W, pady=(12, 0))

    def _status_text(self):
        roi = self.saved_roi_path
        geom = len(getattr(self, "transform_history", []))
        suffix = f" | geometry ops: {geom}" if geom else ""
        if roi and os.path.isfile(roi):
            return f"ROI linked: {os.path.basename(roi)}{suffix}"
        if getattr(self, "background_roi_rect", None) is not None:
            return f"ROI linked from SCAMP geometry sidecar{suffix}"
        return "No matching ROI linked yet."

    def load_full_stack_editor(self):
        """Open a temporary StackEditor preview tab for this CZI.

        The lightweight CZI record stays in the Open Files list. The preview tab
        can be closed after ROI/geometry editing; geometry is copied back to the
        lightweight record and persisted to the SCAMP sidecar file.
        """
        if self.preview_editor is not None and self.preview_editor in self.app.editors:
            self.app.notebook.select(self.preview_editor)
            return
        try:
            self.app._log(f"Loading preview stack for {self.name} ...")
            self.app.update_idletasks()
            stack, orig_dtype, z_step_um = load_microscopy_stack(self.path)

            # Reapply stored geometry so a reopened preview matches the saved
            # ROI coordinate system and the batch background-subtraction input.
            history = list(getattr(self, "transform_history", []))
            if history:
                stack = apply_stack_transform_history(stack, history)

            real = StackEditor(
                self.app.notebook,
                self.path,
                self.app,
                stack,
                orig_dtype=orig_dtype,
                z_step_um=z_step_um,
                name=self.name + " [preview]",
                group=self.group,
            )
            real.saved_roi_path = self.saved_roi_path
            real.source_base = self.source_base
            real.transform_history = history
            real.background_roi_rect = getattr(self, "background_roi_rect", None)
            real.selection_rect = real.background_roi_rect
            real.source_deferred_editor = self
            real.is_czi_preview_editor = True
            self.preview_editor = real

            self.app.editors.append(real)
            self.app.notebook.add(real, text=real.name[:24])
            self.app.notebook.select(real)
            self.app._refresh_file_bar()

            z_note = f", Z-step {z_step_um:g} µm" if z_step_um else ""
            self.app._log(
                f"Loaded temporary preview {real.name} (Z={stack.shape[0]}, {real.w}x{real.h}{z_note}). "
                "Close the preview tab after saving ROI to release memory."
            )
        except Exception as exc:
            self.app._log(f"Could not load preview for {self.name}: {exc}")
            messagebox.showerror("Could not load CZI preview", str(exc))

class StackEditor(ttk.Frame):
    """
    Stack/hyperstack preview editor with Z browsing, rectangle selection,
    reusable ROI restore, crop, transforms, and projection tools.
    """

    def __init__(self, master, path, app, stack, orig_dtype=None, z_step_um=None,
                 name=None, group=""):
        super().__init__(master)
        self.app = app
        self.path = path
        self.name = name if name else os.path.basename(path)
        self.stack = np.asarray(stack, dtype=np.float32)
        self.orig_dtype = orig_dtype
        self.max_val = dtype_max(orig_dtype)
        self.z_step_um = z_step_um
        self.group = group or ""
        self.source_base = os.path.splitext(self.name)[0]
        self.generated_preview = False
        self.is_stack_editor = True
        self.geometry_matches_source = True
        # Transform operations applied in the preview editor. Batch background
        # subtraction replays these on the original CZI stack before applying
        # the ROI, so the processed projection matches the displayed geometry.
        self.transform_history = []
        self.background_roi_rect = None
        self.saved_roi_path = None
        if os.path.splitext(self.path)[1].lower() == ".czi":
            candidate_roi = os.path.splitext(self.path)[0] + ".roi"
            if os.path.isfile(candidate_roi):
                self.saved_roi_path = candidate_roi
            history, rect = load_geometry_sidecar(self.path)
            if history:
                self.transform_history = history
            if rect is not None:
                self.background_roi_rect = rect

        self.z_index = 0
        self.selection_rect = None
        self._rect_artist = None
        self._drag_start = None
        self._drag_kind = None
        self._drag_offset = (0.0, 0.0)
        self._select_mode = False

        self._build_ui()
        self._connect_events()
        self.redraw()

    @property
    def h(self):
        return int(self.stack.shape[1])

    @property
    def w(self):
        return int(self.stack.shape[2])

    def _build_ui(self):
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        self.tools_btn = ttk.Menubutton(toolbar_frame, text="Tools ▾")
        self.tools_menu = tk.Menu(self.tools_btn, tearoff=False)
        self.tools_btn["menu"] = self.tools_menu

        self.tools_menu.add_command(label="Rectangle tool", command=self.enable_rectangle_tool)
        self.tools_menu.add_command(label="Save ROI", command=self.save_roi_to_matching_file)
        self.tools_menu.add_command(label="Save selection for restore", command=self.save_selection_template)
        self.tools_menu.add_command(label="Restore saved selection", command=self.restore_selection_template)
        self.tools_menu.add_separator()
        self.tools_menu.add_command(label="Crop to rectangle", command=self.crop_to_selection)
        self.tools_menu.add_separator()
        self.tools_menu.add_command(label="Flip horizontal", command=self.flip_horizontal)
        self.tools_menu.add_command(label="Flip vertical", command=self.flip_vertical)
        self.tools_menu.add_command(label="Rotate with preview…", command=self.open_rotation_preview)
        self.tools_menu.add_separator()
        self.tools_menu.add_command(label="Projection: mean", command=lambda: self.make_projection("mean"))
        self.tools_menu.add_command(label="Projection: sum", command=lambda: self.make_projection("sum"))
        self.tools_menu.add_command(label="Projection: Z-depth normalised sum", command=lambda: self.make_projection("zdepth"))
        self.tools_btn.pack(side=tk.LEFT, padx=3, pady=2)

        self.status_var = tk.StringVar(value=self._status_text())
        ttk.Label(toolbar_frame, textvariable=self.status_var).pack(side=tk.RIGHT, padx=6)

        slider_frame = ttk.Frame(self)
        slider_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(slider_frame, text="Z").pack(side=tk.LEFT, padx=(6, 2))
        self.z_var = tk.IntVar(value=0)
        self.z_slider = ttk.Scale(
            slider_frame, from_=0, to=max(0, self.stack.shape[0] - 1),
            orient=tk.HORIZONTAL, command=self._on_z_slider)
        self.z_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.z_label_var = tk.StringVar(value=self._z_label())
        ttk.Label(slider_frame, textvariable=self.z_label_var, width=12).pack(side=tk.RIGHT, padx=6)

        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        nav = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        nav.update()
        nav.pack(side=tk.BOTTOM, fill=tk.X)

    def _connect_events(self):
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

    def sync_geometry_to_deferred_source(self):
        """Copy preview geometry back to its lightweight CZI record and sidecar."""
        source = getattr(self, "source_deferred_editor", None)
        if source is None:
            return
        source.transform_history = list(getattr(self, "transform_history", []))
        source.background_roi_rect = getattr(self, "background_roi_rect", None)
        source.saved_roi_path = getattr(self, "saved_roi_path", None)
        source.z_step_um = getattr(self, "z_step_um", None)
        try:
            save_geometry_sidecar(source.path, source.transform_history, source.background_roi_rect)
        except Exception as exc:
            self.app._log(f"Could not save geometry sidecar for {source.name}: {exc}")
        try:
            if hasattr(source, "status_var"):
                source.status_var.set(source._status_text())
        except Exception:
            pass

    def _status_text(self):
        ztxt = f"Z slices: {self.stack.shape[0]}"
        if self.z_step_um is not None:
            ztxt += f" | Z-step: {self.z_step_um:g} µm"
        if self.selection_rect is not None:
            x0, y0, x1, y1 = self.selection_rect
            ztxt += f" | ROI: {abs(int(x1-x0))}x{abs(int(y1-y0))}"
        return ztxt

    def _z_label(self):
        return f"{self.z_index + 1}/{self.stack.shape[0]}"

    def _on_z_slider(self, value):
        try:
            self.z_index = int(round(float(value)))
        except Exception:
            self.z_index = 0
        self.z_index = max(0, min(self.z_index, self.stack.shape[0] - 1))
        self.z_label_var.set(self._z_label())
        self.redraw()

    def current_slice(self):
        return self.stack[self.z_index]

    def redraw(self):
        # Keep the current zoom/pan when redrawing overlays or changing tools.
        # Z-slider changes should update the image content without snapping the
        # view back to the full frame.
        preserve_view = bool(self.ax.images)
        old_xlim = self.ax.get_xlim() if preserve_view else None
        old_ylim = self.ax.get_ylim() if preserve_view else None

        self.ax.clear()
        self.ax.imshow(normalize_display_image(self.current_slice()), cmap="gray", origin="upper")
        self.ax.set_title(self.name, fontsize=9)
        if preserve_view:
            self.ax.set_xlim(old_xlim)
            self.ax.set_ylim(old_ylim)
        else:
            self.ax.set_xlim(0, self.w)
            self.ax.set_ylim(self.h, 0)
        if self.selection_rect is not None:
            x0, y0, x1, y1 = self.selection_rect
            from matplotlib.patches import Rectangle
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                             edgecolor="yellow", linewidth=1.5)
            self.ax.add_patch(rect)
        self.ax.set_axis_off()
        self.status_var.set(self._status_text())
        self.canvas.draw_idle()

    def _normalized_selection(self):
        """Return the active rectangle as ordered coordinates, or None."""
        if self.selection_rect is None:
            return None
        x0, y0, x1, y1 = self.selection_rect
        xa, xb = sorted((float(x0), float(x1)))
        ya, yb = sorted((float(y0), float(y1)))
        return xa, ya, xb, yb

    def _point_in_selection(self, x, y):
        """True when an image-coordinate point is inside the active ROI."""
        rect = self._normalized_selection()
        if rect is None:
            return False
        x0, y0, x1, y1 = rect
        return x0 <= x <= x1 and y0 <= y <= y1

    def _clamp_selection_to_image(self):
        """Clamp the active rectangle into image bounds."""
        if self.selection_rect is None:
            return
        x0, y0, x1, y1 = self.selection_rect
        self.selection_rect = (
            max(0.0, min(float(self.w), float(x0))),
            max(0.0, min(float(self.h), float(y0))),
            max(0.0, min(float(self.w), float(x1))),
            max(0.0, min(float(self.h), float(y1))),
        )

    def enable_rectangle_tool(self):
        self._select_mode = True
        self._drag_start = None
        self._drag_kind = None
        self.status_var.set(
            "Rectangle tool active: drag to create one ROI, drag inside it to move it, click outside to replace it."
        )

    def on_press(self, event):
        if not self._select_mode or event.inaxes != self.ax or event.xdata is None:
            return
        if self.fig.canvas.toolbar is not None and getattr(self.fig.canvas.toolbar, "mode", ""):
            return
        if event.button != 1:
            return
        x = max(0.0, min(float(self.w), float(event.xdata)))
        y = max(0.0, min(float(self.h), float(event.ydata)))
        if self.selection_rect is not None and self._point_in_selection(x, y):
            # Move the single active ROI.
            x0, y0, x1, y1 = self.selection_rect
            self._drag_kind = "move"
            self._drag_start = (x, y)
            self._drag_offset = (x - x0, y - y0)
        else:
            # Replace any existing ROI with a new one.
            self.selection_rect = None
            self._drag_kind = "draw"
            self._drag_start = (x, y)
            self.selection_rect = (x, y, x, y)
        self.redraw()

    def on_motion(self, event):
        if self._drag_start is None or event.inaxes != self.ax or event.xdata is None:
            return
        x = max(0.0, min(float(self.w), float(event.xdata)))
        y = max(0.0, min(float(self.h), float(event.ydata)))
        if self._drag_kind == "move" and self.selection_rect is not None:
            x0, y0, x1, y1 = self.selection_rect
            width = x1 - x0
            height = y1 - y0
            off_x, off_y = self._drag_offset
            new_x0 = x - off_x
            new_y0 = y - off_y
            new_x1 = new_x0 + width
            new_y1 = new_y0 + height
            # Keep the ROI inside the image while preserving its size.
            if new_x0 < 0:
                new_x1 -= new_x0
                new_x0 = 0.0
            if new_y0 < 0:
                new_y1 -= new_y0
                new_y0 = 0.0
            if new_x1 > self.w:
                shift = new_x1 - self.w
                new_x0 -= shift
                new_x1 = float(self.w)
            if new_y1 > self.h:
                shift = new_y1 - self.h
                new_y0 -= shift
                new_y1 = float(self.h)
            self.selection_rect = (new_x0, new_y0, new_x1, new_y1)
        elif self._drag_kind == "draw":
            x0, y0 = self._drag_start
            self.selection_rect = (x0, y0, x, y)
        self.redraw()

    def on_release(self, event):
        if self._drag_start is None:
            return
        if self._drag_kind == "draw" and event.xdata is not None and event.ydata is not None:
            x0, y0 = self._drag_start
            x = max(0.0, min(float(self.w), float(event.xdata)))
            y = max(0.0, min(float(self.h), float(event.ydata)))
            self.selection_rect = (x0, y0, x, y)
        self._clamp_selection_to_image()
        # Discard accidental click-only rectangles.
        rect = self._normalized_selection()
        if rect is not None:
            x0, y0, x1, y1 = rect
            if abs(x1 - x0) < 2 or abs(y1 - y0) < 2:
                self.selection_rect = None
        self._drag_start = None
        self._drag_kind = None
        self.redraw()

    def save_roi_to_matching_file(self):
        """Save the active rectangle as an ImageJ .roi next to the source CZI.

        The ROI is written with the same base name as the original CZI:
        /path/sample.czi -> /path/sample.roi. Coordinates are saved in the
        original source image coordinate system, so the command is intentionally
        limited to untransformed CZI stacks.
        """
        if self.selection_rect is None:
            messagebox.showinfo("No selection", "Draw a rectangle before saving an ROI.")
            return

        source_ext = os.path.splitext(self.path)[1].lower()
        if source_ext != ".czi":
            messagebox.showinfo(
                "Save ROI unavailable",
                "Save ROI is available only for stacks opened directly from a CZI file."
            )
            return

        rect = self._normalized_selection()
        if rect is None:
            messagebox.showinfo("No selection", "Draw a rectangle before saving an ROI.")
            return
        x0, y0, x1, y1 = rect
        if abs(x1 - x0) < 2 or abs(y1 - y0) < 2:
            messagebox.showinfo("Selection too small", "Draw a larger rectangle before saving an ROI.")
            return

        if not _HAVE_ROIFILE:
            why = _READER_IMPORT_ERRORS.get("roifile", "package not found")
            messagebox.showerror(
                "roifile missing",
                "Saving ImageJ ROI files requires the roifile package, which "
                "could not be imported in the Python running SCAMP "
                f"({sys.executable}).\n\nImport error: {why}\n\n"
                "Install it in the SAME environment you launch SCAMP from:\n"
                "    pip install roifile\n"
                "then restart SCAMP. (Run scamp_doctor.py to confirm.)"
            )
            return

        roi_path = os.path.splitext(self.path)[0] + ".roi"
        xa, ya, xb, yb = [int(round(v)) for v in (x0, y0, x1, y1)]
        xa = max(0, min(self.w, xa))
        xb = max(0, min(self.w, xb))
        ya = max(0, min(self.h, ya))
        yb = max(0, min(self.h, yb))
        if xb <= xa or yb <= ya:
            messagebox.showinfo("Invalid selection", "The ROI rectangle is outside the image bounds.")
            return

        try:
            # A polygon ROI with four corners is broadly compatible with ImageJ/Fiji
            # and works as a filled background mask when reloaded by roifile.
            points = np.array([
                [xa, ya],
                [xb, ya],
                [xb, yb],
                [xa, yb],
            ], dtype=np.int16)
            roi = ImagejRoi.frompoints(points, name=os.path.basename(roi_path))
            roi.tofile(roi_path)
        except Exception as exc:
            messagebox.showerror("Save ROI failed", str(exc))
            return

        self.saved_roi_path = roi_path
        self.app._refresh_file_bar()
        self.app._log(f"Saved ROI → {roi_path}")
        messagebox.showinfo("ROI saved", f"Saved ROI:\n{roi_path}")

    def save_selection_template(self):
        if self.selection_rect is None:
            messagebox.showinfo("No selection", "Draw a rectangle first.")
            return
        x0, y0, x1, y1 = self.selection_rect
        self.app.saved_selection_template = {
            "rect": (float(x0), float(y0), float(x1), float(y1)),
            "shape": (self.h, self.w),
        }
        self.app._log(f"Saved selection template from {self.name}.")

    def restore_selection_template(self):
        template = getattr(self.app, "saved_selection_template", None)
        if not template:
            messagebox.showinfo("No saved selection", "Save a selection first.")
            return
        x0, y0, x1, y1 = template["rect"]
        th, tw = template["shape"]
        sx = self.w / float(tw) if tw else 1.0
        sy = self.h / float(th) if th else 1.0
        self.selection_rect = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
        self.redraw()
        self.app._log(f"Restored selection template on {self.name}.")

    def crop_to_selection(self):
        slices = rectangle_to_slices(self.selection_rect, self.stack.shape[1:])
        if slices is None:
            messagebox.showinfo("No selection", "Draw a rectangle before cropping.")
            return
        ys, xs = slices
        self.stack = self.stack[:, ys, xs].astype(np.float32, copy=False)
        self.transform_history.append(("crop", int(ys.start), int(ys.stop), int(xs.start), int(xs.stop)))
        self.geometry_matches_source = False
        self.selection_rect = None
        self.background_roi_rect = None
        self.z_slider.configure(to=max(0, self.stack.shape[0] - 1))
        self.redraw()
        self.app._log(f"Cropped {self.name} to {self.w}x{self.h}.")

    def flip_horizontal(self):
        self.stack = self.stack[:, :, ::-1].astype(np.float32, copy=False)
        self.transform_history.append(("flip_h",))
        self.geometry_matches_source = False
        self.background_roi_rect = None
        self.sync_geometry_to_deferred_source()
        if self.selection_rect is not None:
            x0, y0, x1, y1 = self.selection_rect
            self.selection_rect = (self.w - x0, y0, self.w - x1, y1)
        self.redraw()
        self.app._log(f"Flipped horizontally: {self.name}")

    def flip_vertical(self):
        self.stack = self.stack[:, ::-1, :].astype(np.float32, copy=False)
        self.transform_history.append(("flip_v",))
        self.geometry_matches_source = False
        self.background_roi_rect = None
        self.sync_geometry_to_deferred_source()
        if self.selection_rect is not None:
            x0, y0, x1, y1 = self.selection_rect
            self.selection_rect = (x0, self.h - y0, x1, self.h - y1)
        self.redraw()
        self.app._log(f"Flipped vertically: {self.name}")

    def _rotate_stack(self, angle):
        """Rotate the stack while preserving the exact original ZYX shape.

        The image data are rotated around the center of each YX plane and
        sampled back into the same canvas. This intentionally clips corners
        when needed, because downstream projection and straightening expect
        the original image dimensions to stay unchanged.
        """
        original_shape = self.stack.shape
        try:
            from scipy import ndimage as ndi
            rotated = ndi.rotate(self.stack, angle=float(angle), axes=(1, 2),
                                 reshape=False, order=1, mode="constant",
                                 cval=0.0, prefilter=False)
        except Exception:
            from skimage.transform import rotate as sk_rotate
            rotated = [sk_rotate(sl, angle=float(angle), resize=False, order=1,
                                 mode="constant", cval=0.0, preserve_range=True)
                       for sl in self.stack]
            rotated = np.asarray(rotated, dtype=np.float64)

        if rotated.shape != original_shape:
            fixed = np.zeros(original_shape, dtype=np.float64)
            z = min(original_shape[0], rotated.shape[0])
            y = min(original_shape[1], rotated.shape[1])
            x = min(original_shape[2], rotated.shape[2])
            fixed[:z, :y, :x] = rotated[:z, :y, :x]
            rotated = fixed
        return np.asarray(rotated, dtype=np.float32)

    def open_rotation_preview(self):
        dlg = tk.Toplevel(self)
        dlg.title("Rotate with preview")
        dlg.transient(self)
        dlg.resizable(True, True)
        angle_var = tk.DoubleVar(value=0.0)

        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=dlg)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        def draw_preview(*_):
            angle = angle_var.get()
            try:
                from scipy import ndimage as ndi
                prev = ndi.rotate(self.current_slice(), angle=angle, reshape=False,
                                  order=1, mode="constant", cval=0.0,
                                  prefilter=False)
            except Exception:
                from skimage.transform import rotate as sk_rotate
                prev = sk_rotate(self.current_slice(), angle=angle, resize=False,
                                 order=1, mode="constant", cval=0.0,
                                 preserve_range=True)
            ax.clear()
            ax.imshow(normalize_display_image(prev), cmap="gray", origin="upper")
            ax.set_title(f"Rotation: {angle:.1f}°", fontsize=9)
            ax.set_axis_off()
            canvas.draw_idle()

        slider = ttk.Scale(dlg, from_=-180, to=180, orient=tk.HORIZONTAL,
                           variable=angle_var, command=lambda v: draw_preview())
        slider.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        btns = ttk.Frame(dlg)
        btns.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=8)

        def apply_rotation():
            angle = angle_var.get()
            before_shape = self.stack.shape
            self.stack = self._rotate_stack(angle).astype(np.float32, copy=False)
            self.transform_history.append(("rotate", float(angle)))
            self.geometry_matches_source = False
            after_shape = self.stack.shape
            self.selection_rect = None
            self.background_roi_rect = None
            self.sync_geometry_to_deferred_source()
            self.z_slider.configure(to=max(0, self.stack.shape[0] - 1))
            self.redraw()
            self.app._log(
                f"Rotated {self.name} by {angle:.1f} degrees "
                f"(shape preserved: {before_shape} -> {after_shape}).")
            dlg.destroy()

        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side=tk.LEFT)
        ttk.Button(btns, text="Apply", command=apply_rotation).pack(side=tk.RIGHT)
        draw_preview()

    def _ask_z_step(self):
        if self.z_step_um is not None and self.z_step_um > 0:
            return self.z_step_um
        from tkinter import simpledialog
        value = simpledialog.askfloat(
            "Z-step required",
            "Enter Z-step size in µm for Z-depth normalised sum:",
            minvalue=0.000001,
            parent=self,
        )
        if value is not None:
            self.z_step_um = float(value)
        return self.z_step_um

    def make_projection(self, method):
        if method == "zdepth":
            z_step = self._ask_z_step()
            if z_step is None:
                return
        else:
            z_step = self.z_step_um
        try:
            proj = project_stack(self.stack, method=method,
                                 background_rect=self.selection_rect,
                                 z_step_um=z_step)
        except Exception as exc:
            messagebox.showerror("Projection failed", str(exc))
            return

        if not self.app.experiment_dir:
            messagebox.showwarning("No experiment", "No ExperimentID is set.")
            return
        norm_dir = os.path.join(self.app.experiment_dir, "normalized")
        try:
            os.makedirs(norm_dir, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Output folder error", str(exc))
            return

        kind = {"mean": "meanIP", "sum": "sumIP", "zdepth": "Zdepth_normalised_sumIP"}[method]
        out_ext = projection_extension_for_source(self.path)
        out_name = build_filename(self.source_base, self.app.experiment_id,
                                  self.group, kind=kind, ext=out_ext)
        out_path = os.path.join(norm_dir, out_name)
        try:
            # Keep generated projection TIFFs in a consistent SCAMP storage unit.
            # Z-depth projections often contain native fractional values; saving
            # them directly as uint16 would threshold sub-1 signal to zero.
            storage_scale = 1.0
            save_proj = proj
            if method == "zdepth":
                storage_scale = float(SCAMP_ZDEPTH_UINT16_SCALE)
                save_proj = np.asarray(proj, dtype=np.float64) * storage_scale
            else:
                save_proj, extra_scale, _note = _scamp_storage_array_for_uint16(proj, input_storage_scale=1.0)
                storage_scale = float(extra_scale)
            save_image16(out_path, save_proj)
            ed = SampleEditor(self.app.notebook, out_path, self.app,
                              image_array=save_proj, name=out_name, group=self.group)
            ed.source_base = self.source_base
            ed.normalized = save_proj
            ed.storage_scale_factor = storage_scale
            ed.storage_unit_note = (
                f"Projection is stored in SCAMP storage units: native intensity ×{storage_scale:g}."
                if storage_scale != 1.0 else "Projection is stored in native units."
            )
            self.app.editors.append(ed)
            self.app.notebook.add(ed, text=ed.name[:24])
            self.app.notebook.select(ed)
            self.app._refresh_file_bar()
            self.app._log(f"Created projection → {out_name} (storage scale ×{storage_scale:g})")
        except Exception as exc:
            self.app._log(f"Projection created, but could not save/open: {exc}")


def apply_stack_transform_history(stack, history):
    """Replay preview-editor geometry operations on a freshly loaded stack.

    This is used by batch background subtraction so a CZI that was flipped,
    rotated, or cropped before ROI placement is processed in the same geometry
    that the user saw when drawing the ROI.
    """
    out = np.asarray(stack, dtype=np.float32)
    for op in history or []:
        name = op[0]
        if name == "flip_h":
            out = out[:, :, ::-1]
        elif name == "flip_v":
            out = out[:, ::-1, :]
        elif name == "crop":
            _, y0, y1, x0, x1 = op
            out = out[:, int(y0):int(y1), int(x0):int(x1)]
        elif name == "rotate":
            angle = float(op[1])
            try:
                from scipy import ndimage as ndi
                out = ndi.rotate(out, angle=angle, axes=(1, 2), reshape=False,
                                 order=1, mode="constant", cval=0.0,
                                 prefilter=False)
            except Exception:
                from skimage.transform import rotate as sk_rotate
                out = np.asarray([
                    sk_rotate(sl, angle=angle, resize=False, order=1,
                              mode="constant", cval=0.0, preserve_range=True)
                    for sl in out
                ], dtype=np.float32)
        else:
            raise ValueError(f"Unknown geometry operation: {name}")
        out = np.asarray(out, dtype=np.float32)
    return out


def rectangle_mask_from_rect(rect, shape):
    """Create a boolean mask from a rectangle in the current image geometry."""
    slices = rectangle_to_slices(rect, shape)
    if slices is None:
        raise ValueError("Saved ROI rectangle is empty or outside the image.")
    ys, xs = slices
    mask = np.zeros(shape, dtype=bool)
    mask[ys, xs] = True
    return mask

# ======================================================================
#  Modern UI theme helpers
# ======================================================================
def apply_modern_theme(root):
    """Apply the SCAMP dark UI theme.

    The app still uses standard Tk/ttk widgets so the scientific workflow is
    unchanged. When ttkbootstrap is installed, the Darkly theme is used. If it
    is missing, the app falls back to a small built-in dark ttk style instead
    of failing to start.
    """
    palette = {
        "bg": "#111827",
        "panel": "#17212f",
        "panel2": "#1f2937",
        "surface": "#0f172a",
        "text": "#e5e7eb",
        "muted": "#9ca3af",
        "border": "#334155",
        "primary": "#0d6efd",
        "success": "#198754",
        "warning": "#f59e0b",
        "danger": "#dc3545",
    }
    root._scamp_palette = palette
    root._scamp_dark_ui = True

    try:
        if _HAVE_TTKBOOTSTRAP:
            root._scamp_bootstrap_style = ttkbs.Style(theme="darkly")
            style = root._scamp_bootstrap_style
        else:
            style = ttk.Style(root)
            try:
                style.theme_use("clam")
            except Exception:
                pass
        root._scamp_style = style
    except Exception:
        style = ttk.Style(root)
        root._scamp_style = style

    try:
        root.configure(bg=palette["bg"])
    except Exception:
        pass

    def cfg(name, **kwargs):
        try:
            style.configure(name, **kwargs)
        except Exception:
            pass

    def map_style(name, **kwargs):
        try:
            style.map(name, **kwargs)
        except Exception:
            pass

    cfg("TFrame", background=palette["bg"])
    cfg("SCAMP.Toolbar.TFrame", background=palette["surface"])
    cfg("SCAMP.Sidebar.TFrame", background=palette["panel"])
    cfg("SCAMP.Card.TFrame", background=palette["panel2"], relief="flat")
    cfg("SCAMP.Actions.TFrame", background=palette["panel"])
    cfg("TLabel", background=palette["bg"], foreground=palette["text"])
    cfg("SCAMP.SidebarTitle.TLabel", background=palette["panel"], foreground=palette["text"], font=("TkDefaultFont", 10, "bold"))
    cfg("SCAMP.Muted.TLabel", background=palette["panel2"], foreground=palette["muted"])
    cfg("SCAMP.Success.TLabel", background=palette["panel2"], foreground="#75d39b")
    cfg("SCAMP.Warning.TLabel", background=palette["panel2"], foreground=palette["warning"])
    cfg("TNotebook", background=palette["bg"], borderwidth=0)
    cfg("TNotebook.Tab", padding=(12, 7), font=("TkDefaultFont", 9))
    cfg("TPanedwindow", background=palette["bg"])
    cfg("Vertical.TScrollbar", background=palette["panel2"], troughcolor=palette["surface"], bordercolor=palette["surface"], arrowcolor=palette["text"])

    # Bootstrap theme style names. If ttkbootstrap is missing these still work
    # because we configure them manually.
    for name, color in (("primary", palette["primary"]), ("success", palette["success"]), ("warning", palette["warning"]), ("danger", palette["danger"])):
        cfg(f"{name}.TButton", background=color, foreground="#ffffff", borderwidth=0, focusthickness=0, padding=(12, 7))
        map_style(f"{name}.TButton", background=[("active", color), ("pressed", color)], foreground=[("disabled", "#9ca3af")])
    cfg("secondary.TButton", background=palette["panel2"], foreground=palette["text"], borderwidth=0, padding=(10, 6))
    map_style("secondary.TButton", background=[("active", "#374151"), ("pressed", "#374151")])
    cfg("SCAMP.File.TButton", background=palette["panel2"], foreground=palette["text"], anchor="w", padding=(8, 5), borderwidth=0)
    map_style("SCAMP.File.TButton", background=[("active", "#334155"), ("pressed", "#334155")])

    return palette

# ======================================================================
#  Main application window
# ======================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.palette = apply_modern_theme(self)
        self.title("SCAMP — Spinal Calcification & Mineralization Profiler")
        self.geometry("1180x780")
        self.minsize(980, 650)

        self.editors = []  # list[SampleEditor]

        # experiment state (set by the startup dialogs)
        self.experiment_id = None
        self.experiment_dir = None
        self.parent_dir = None
        self.conditions = ["control"]

        self._build_ui()
        self._report_optional_deps()

        # Run the startup dialogs after the window is shown.
        self.after(50, self._startup_sequence)

    # ------------------------------------------------------------------
    def _startup_sequence(self):
        """Establish ExperimentID, create its directory, then set up
        conditions. If the user cancels the ExperimentID step, the app
        closes (nothing can be saved without an experiment directory)."""
        ok = self._ask_experiment_id()
        if not ok:
            self._log("No ExperimentID set — closing.")
            self.destroy()
            return
        self._ask_conditions()
        self._log(f"ExperimentID: {self.experiment_id}")
        self._log(f"Experiment directory: {self.experiment_dir}")
        self._log("Conditions: " + ", ".join(self.conditions))
        self._refresh_file_bar()

    def _ask_experiment_id(self):
        """Modal dialog to choose/confirm the ExperimentID and parent folder.
        Returns True if an experiment directory was created."""
        parent = filedialog.askdirectory(
            title="Choose the folder that will contain this experiment")
        if not parent:
            return False
        self.parent_dir = parent

        dlg = tk.Toplevel(self)
        dlg.title("Establish ExperimentID")
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)

        result = {"ok": False}

        ttk.Label(dlg, text="ExperimentID (a directory with this name will be "
                            "created):").pack(anchor=tk.W, padx=12, pady=(12, 4))

        id_var = tk.StringVar(value=propose_experiment_id(parent))
        entry = ttk.Entry(dlg, textvariable=id_var, width=40)
        entry.pack(fill=tk.X, padx=12)

        status = tk.StringVar(value="")
        status_lbl = ttk.Label(dlg, textvariable=status, foreground="#c0392b")
        status_lbl.pack(anchor=tk.W, padx=12, pady=(2, 0))

        in_parent = os.path.basename(parent.rstrip(os.sep)) or parent
        ttk.Label(dlg, text=f"Inside: {parent}",
                  foreground="#666").pack(anchor=tk.W, padx=12, pady=(2, 8))

        def regenerate():
            id_var.set(propose_experiment_id(parent))
            status.set("")

        def confirm():
            eid = _sanitize(id_var.get())
            if not eid:
                status.set("Please enter a non-empty ID.")
                return
            if not experiment_id_is_free(parent, eid):
                status.set("That ID already exists here — pick another.")
                return
            exp_dir = os.path.join(parent, eid)
            try:
                os.makedirs(exp_dir, exist_ok=False)
            except FileExistsError:
                status.set("That ID already exists here — pick another.")
                return
            except Exception as exc:
                status.set(f"Could not create directory: {exc}")
                return
            self.experiment_id = eid
            self.experiment_dir = exp_dir
            result["ok"] = True
            dlg.destroy()

        btns = ttk.Frame(dlg)
        btns.pack(fill=tk.X, padx=12, pady=(0, 12))
        ttk.Button(btns, text="Propose another",
                   command=regenerate).pack(side=tk.LEFT)
        ttk.Button(btns, text="Create",
                   command=confirm).pack(side=tk.RIGHT)

        entry.focus_set()
        dlg.bind("<Return>", lambda e: confirm())
        self.wait_window(dlg)
        return result["ok"]

    def _ask_conditions(self):
        """Modal dialog to set up the experiment's conditions. Starts with an
        editable 'control' default; rows can be added with 'Add more'."""
        dlg = tk.Toplevel(self)
        dlg.title("Set up conditions")
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)

        ttk.Label(dlg, text="Conditions for this experiment:").pack(
            anchor=tk.W, padx=12, pady=(12, 4))

        rows_frame = ttk.Frame(dlg)
        rows_frame.pack(fill=tk.BOTH, padx=12)

        row_vars = []

        def add_row(value=""):
            row = ttk.Frame(rows_frame)
            row.pack(fill=tk.X, pady=2)
            var = tk.StringVar(value=value)
            ttk.Entry(row, textvariable=var, width=32).pack(
                side=tk.LEFT, fill=tk.X, expand=True)

            def remove():
                if len(row_vars) <= 1:
                    return  # keep at least one
                row_vars.remove(entry_pair)
                row.destroy()

            tk.Button(row, text="x", width=2, fg="#c0392b",
                      activeforeground="#e74c3c",
                      font=("TkDefaultFont", 10, "bold"),
                      relief=tk.FLAT, bd=0, cursor="hand2",
                      command=remove).pack(side=tk.RIGHT, padx=(4, 0))
            entry_pair = var
            row_vars.append(entry_pair)

        # start with the existing conditions (default ["control"])
        for c in (self.conditions or ["control"]):
            add_row(c)

        status = tk.StringVar(value="")
        ttk.Label(dlg, textvariable=status, foreground="#c0392b").pack(
            anchor=tk.W, padx=12, pady=(2, 0))

        def confirm():
            seen = []
            for var in row_vars:
                name = var.get().strip()
                if name and name not in seen:
                    seen.append(name)
            if not seen:
                status.set("Enter at least one condition.")
                return
            self.conditions = seen
            dlg.destroy()

        btns = ttk.Frame(dlg)
        btns.pack(fill=tk.X, padx=12, pady=(8, 12))
        ttk.Button(btns, text="Add more",
                   command=lambda: add_row("")).pack(side=tk.LEFT)
        ttk.Button(btns, text="Done", command=confirm).pack(side=tk.RIGHT)

        self.wait_window(dlg)

    # ------------------------------------------------------------------
    def _build_ui(self):
        top = ttk.Frame(self, style="SCAMP.Toolbar.TFrame")
        top.pack(side=tk.TOP, fill=tk.X)

        primary = ttk.Frame(top, style="SCAMP.Toolbar.TFrame")
        primary.pack(side=tk.LEFT, fill=tk.X, padx=8, pady=8)
        ttk.Button(primary, text="Open", style="primary.TButton",
                   command=self.add_images).pack(side=tk.LEFT, padx=4)
        ttk.Button(primary, text="Add straighten guides", style="secondary.TButton",
                   command=self.add_straighten_guides).pack(side=tk.LEFT, padx=4)
        ttk.Button(primary, text="Straighten current", style="success.TButton",
                   command=self.straighten_current).pack(side=tk.LEFT, padx=4)

        batch = ttk.Frame(top, style="SCAMP.Toolbar.TFrame")
        batch.pack(side=tk.RIGHT, padx=8, pady=8)
        ttk.Button(batch, text="Process all - heatmap + tables", style="warning.TButton",
                   command=self.process_all).pack(side=tk.RIGHT, padx=4)

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.file_bar = ttk.Frame(main, width=300, style="SCAMP.Sidebar.TFrame")
        self.file_bar.pack_propagate(False)
        ttk.Label(self.file_bar, text="Open files", style="SCAMP.SidebarTitle.TLabel").pack(
            side=tk.TOP, anchor=tk.W, padx=10, pady=(10, 4))
        ttk.Label(self.file_bar, text="name / condition", style="SCAMP.SidebarTitle.TLabel").pack(
            side=tk.TOP, anchor=tk.W, padx=10, pady=(0, 8))

        # Scrollable file list. The batch/action buttons live outside this
        # canvas, so they remain visible even when the list is long.
        self.file_list_container = ttk.Frame(self.file_bar, style="SCAMP.Sidebar.TFrame")
        self.file_list_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.file_list_canvas = tk.Canvas(
            self.file_list_container, highlightthickness=0, borderwidth=0,
            bg=self.palette.get("panel", "#17212f"), bd=0
        )
        self.file_list_scrollbar = ttk.Scrollbar(
            self.file_list_container, orient=tk.VERTICAL,
            command=self.file_list_canvas.yview
        )
        self.file_list_canvas.configure(yscrollcommand=self.file_list_scrollbar.set)
        self.file_list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.file_list = ttk.Frame(self.file_list_canvas)
        self.file_list_window = self.file_list_canvas.create_window(
            (0, 0), window=self.file_list, anchor="nw"
        )

        def _sync_file_list_scroll_region(event=None):
            self.file_list_canvas.configure(
                scrollregion=self.file_list_canvas.bbox("all")
            )
            try:
                needed = (self.file_list.winfo_reqheight() > self.file_list_canvas.winfo_height() or len(getattr(self, "editors", [])) > 6)
                if needed and not self.file_list_scrollbar.winfo_ismapped():
                    self.file_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, before=self.file_list_canvas)
                elif not needed and self.file_list_scrollbar.winfo_ismapped():
                    self.file_list_scrollbar.pack_forget()
                    self.file_list_canvas.yview_moveto(0)
            except Exception:
                pass

        def _sync_file_list_width(event):
            self.file_list_canvas.itemconfigure(
                self.file_list_window, width=event.width
            )
            _sync_file_list_scroll_region()

        def _on_file_list_mousewheel(event):
            if not self.file_list_scrollbar.winfo_ismapped():
                return
            delta = event.delta
            if delta == 0 and hasattr(event, "num"):
                delta = 120 if event.num == 4 else -120
            self.file_list_canvas.yview_scroll(int(-1 * (delta / 120)), "units")

        self.file_list.bind("<Configure>", _sync_file_list_scroll_region)
        self.file_list_canvas.bind("<Configure>", _sync_file_list_width)
        self.file_list_canvas.bind("<MouseWheel>", _on_file_list_mousewheel)
        self.file_list.bind("<MouseWheel>", _on_file_list_mousewheel)
        self.file_list_canvas.bind("<Button-4>", _on_file_list_mousewheel)
        self.file_list_canvas.bind("<Button-5>", _on_file_list_mousewheel)
        self.file_list.bind("<Button-4>", _on_file_list_mousewheel)
        self.file_list.bind("<Button-5>", _on_file_list_mousewheel)

        # Fixed action bar: not part of the scrollable file list.
        self.file_actions = ttk.Frame(self.file_bar, style="SCAMP.Actions.TFrame")
        self.file_actions.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        ttk.Button(
            self.file_actions,
            text="Subtract background",
            style="danger.TButton",
            command=self.subtract_background_for_czi_with_rois,
        ).pack(side=tk.TOP, fill=tk.X)

        self.notebook = ttk.Notebook(main)
        main.add(self.file_bar, weight=0)
        main.add(self.notebook, weight=1)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self.log = tk.Text(
            self, height=7, wrap="word",
            bg=self.palette.get("surface", "#0f172a"),
            fg=self.palette.get("text", "#e5e7eb"),
            insertbackground=self.palette.get("text", "#e5e7eb"),
            relief=tk.FLAT, bd=0, padx=10, pady=8
        )
        self.log.pack(side=tk.BOTTOM, fill=tk.X)
        self._log("Starting up — set the ExperimentID and conditions.")

    def _report_optional_deps(self):
        if not _HAVE_PIL and not _HAVE_CV2 and not _HAVE_TIFFFILE:
            self._log("WARNING: none of tifffile, Pillow, or OpenCV found — "
                      "cannot read or write images. Install one of them.")
        elif not _HAVE_TIFFFILE:
            self._log("Note: tifffile not found — TIFFs handled via Pillow/"
                      "OpenCV (fine, but tifffile is recommended).")
        if not _HAVE_PANDAS:
            self._log("Note: pandas not found — XLSX export disabled, CSV only.")

        # CZI reader status — report up front so a broken reader install is
        # obvious before the user tries to open a file.
        if _CZI_READER is not None:
            self._log(f"CZI reader: {_CZI_READER_NAME} "
                      f"(Python: {os.path.basename(sys.executable)}).")
        else:
            self._log("WARNING: no CZI reader could be imported in this "
                      f"environment ({sys.executable}). 'Import CZI' / opening "
                      ".czi will fail until you install bioio + bioio-czi "
                      "(or aicsimageio + aicspylibczi) here.")
            for name, err in _READER_IMPORT_ERRORS.items():
                self._log(f"    {name} import failed -> {err}")

        if not _HAVE_ROIFILE:
            why = _READER_IMPORT_ERRORS.get("roifile", "package not found")
            self._log("WARNING: roifile not importable here — saving/reading "
                      f"ImageJ ROIs will fail ({why}). Install with: "
                      "pip install roifile")

    def _log(self, msg):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.update_idletasks()

    # ------------------------------------------------------------------
    def _refresh_file_bar(self):
        """Rebuild the file sidebar with a name button, condition entry, and
        close button for each open image."""
        for child in self.file_list.winfo_children():
            child.destroy()
        for ed in self.editors:
            row = ttk.Frame(self.file_list, style="SCAMP.Card.TFrame")
            row.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

            top_line = ttk.Frame(row, style="SCAMP.Card.TFrame")
            top_line.pack(side=tk.TOP, fill=tk.X)
            label = ed.name[:24] + ("..." if len(ed.name) > 24 else "")
            if ed.generated_preview:
                label = "↳ " + label
            ttk.Button(top_line, text=label, style="SCAMP.File.TButton",
                       command=lambda e=ed: self._select_editor(e)).pack(
                side=tk.LEFT, fill=tk.X, expand=True)
            # Close button: lowercase "x" in a reddish hue. Use a plain
            # tk.Button because ttk themes often ignore foreground colour.
            close_btn = tk.Button(
                top_line, text="x", width=2,
                fg="#ff6b6b", bg=self.palette.get("panel2", "#1f2937"),
                activeforeground="#ffffff", activebackground="#dc3545",
                font=("TkDefaultFont", 10, "bold"),
                relief=tk.FLAT, bd=0, padx=4, cursor="hand2",
                command=lambda e=ed: self.close_editor(e))
            close_btn.pack(side=tk.RIGHT, padx=(2, 0))

            # condition / group editor (skip for generated previews)
            if not ed.generated_preview:
                cond_line = ttk.Frame(row, style="SCAMP.Card.TFrame")
                cond_line.pack(side=tk.TOP, fill=tk.X, pady=(1, 3))
                ttk.Label(cond_line, text="cond:", style="SCAMP.Muted.TLabel").pack(side=tk.LEFT)
                # ensure the sample's current group is among the choices
                choices = list(self.conditions)
                if ed.group and ed.group not in choices:
                    choices.append(ed.group)
                var = tk.StringVar(value=ed.group or (choices[0] if choices else ""))
                combo = ttk.Combobox(cond_line, textvariable=var,
                                     values=choices, state="readonly",
                                     width=14)
                combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
                # keep the editor's group in sync with the selection
                def _on_change(*_a, e=ed, v=var):
                    e.group = v.get().strip()
                var.trace_add("write", _on_change)

            if getattr(ed, "is_deferred_czi", False):
                ttk.Button(
                    row,
                    text="Load preview / edit ROI",
                    style="primary.TButton",
                    command=lambda e=ed: e.load_full_stack_editor(),
                ).pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 3))
                if getattr(ed, "preview_editor", None) is not None:
                    ttk.Label(
                        row,
                        text="Preview open (close it to free RAM)",
                        style="SCAMP.Warning.TLabel",
                    ).pack(side=tk.TOP, anchor=tk.W, padx=8, pady=(0, 3))

            saved_roi = getattr(ed, "saved_roi_path", None)
            if saved_roi:
                roi_name = os.path.basename(saved_roi)
                ttk.Label(
                    row,
                    text=f"ROI: {roi_name}",
                    style="SCAMP.Success.TLabel",
                ).pack(side=tk.TOP, anchor=tk.W, padx=8, pady=(0, 3))

    def _select_editor(self, ed):
        """Select an editor from the sidebar if it is still open."""
        if ed in self.editors:
            self.notebook.select(ed)

    def _on_tab_changed(self, event=None):
        """Keep the sidebar available for future selection-state styling."""
        self._refresh_file_bar()

    def close_editor(self, ed):
        """Close one image/editor and release large preview arrays when possible."""
        if ed not in self.editors:
            return

        # Temporary CZI preview tabs are memory-heavy. Before closing one, copy
        # its geometry/ROI state back to the lightweight CZI placeholder and to
        # the sidecar JSON, then explicitly drop the stack array.
        source = getattr(ed, "source_deferred_editor", None)
        if source is not None:
            try:
                ed.sync_geometry_to_deferred_source()
            except Exception as exc:
                self._log(f"Could not sync preview geometry for {ed.name}: {exc}")
            source.preview_editor = None

        self.notebook.forget(ed)
        self.editors.remove(ed)

        if getattr(ed, "preview_editor", None) is not None:
            ed.preview_editor = None
        for other in self.editors:
            if getattr(other, "preview_editor", None) is ed:
                other.preview_editor = None

        # Best-effort release of heavy matplotlib and image data.
        try:
            if hasattr(ed, "canvas"):
                ed.canvas.get_tk_widget().destroy()
        except Exception:
            pass
        try:
            if hasattr(ed, "fig"):
                ed.fig.clear()
        except Exception:
            pass
        try:
            if hasattr(ed, "stack"):
                ed.stack = None
        except Exception:
            pass
        try:
            if hasattr(ed, "img"):
                ed.img = None
        except Exception:
            pass
        gc.collect()

        self._refresh_file_bar()
        self._log(f"Closed {ed.name}")

    def _czi_editors_with_rois(self):
        """Return open CZI stack editors that have a matching assigned ROI file."""
        result = []
        for ed in self.editors:
            if not getattr(ed, "is_stack_editor", False):
                continue
            # Temporary preview tabs copy geometry back to the lightweight CZI
            # record. Process the lightweight record only to avoid duplicates.
            if getattr(ed, "source_deferred_editor", None) is not None:
                continue
            if os.path.splitext(getattr(ed, "path", ""))[1].lower() != ".czi":
                continue
            roi_path = getattr(ed, "saved_roi_path", None)
            roi_rect = getattr(ed, "background_roi_rect", None)
            if not roi_path:
                candidate = os.path.splitext(ed.path)[0] + ".roi"
                if os.path.isfile(candidate):
                    roi_path = candidate
                    ed.saved_roi_path = candidate
            # Loaded/transformed CZI editors can provide an in-memory ROI
            # rectangle in transformed coordinates. Lazy placeholders fall back
            # to the same-name .roi file next to the CZI.
            if roi_rect is not None or (roi_path and os.path.isfile(roi_path)):
                result.append((ed, roi_path))
        return result

    def _ask_fallback_z_step(self):
        """Ask once for a fallback Z-step when CZI metadata does not expose it."""
        from tkinter import simpledialog
        return simpledialog.askfloat(
            "Z-step required",
            "One or more CZI files do not expose a Z-step in metadata.\n"
            "Enter a fallback Z-step size in µm for Z-depth normalised sum:",
            minvalue=0.000001,
            parent=self,
        )

    def subtract_background_for_czi_with_rois(self):
        """Batch-create Z-depth normalised sum projections for open CZI files.

        Only open CZI stack editors with an assigned same-name .roi file are
        processed. The ROI is used as the background region. Results are saved
        into a dedicated folder inside the experiment directory and opened as
        new 2-D projection images.
        """
        if not self.experiment_dir:
            messagebox.showwarning("No experiment", "No ExperimentID is set.")
            return
        pairs = self._czi_editors_with_rois()
        if not pairs:
            messagebox.showinfo(
                "No CZI + ROI pairs",
                "No open CZI files have an assigned same-name .roi file.\n"
                "Open CZI files, draw a rectangle, then use Tools → Save ROI first."
            )
            return
        if not _HAVE_ROIFILE or not _HAVE_SKDRAW:
            messagebox.showerror(
                "ROI support missing",
                "Background subtraction needs roifile and scikit-image. Update the environment and try again."
            )
            return

        out_dir = os.path.join(self.experiment_dir, "background_subtracted")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Output folder error", str(exc))
            return

        fallback_z = None

        created = 0
        skipped = 0
        for ed, roi_path in pairs:
            base = os.path.splitext(os.path.basename(ed.path))[0]
            stack = None
            proj = None
            try:
                # Load one CZI at a time, process it, then release the stack.
                # This keeps peak RAM usage close to one file instead of the
                # total size of all opened CZI files.
                self._log(f"Background subtraction: loading {base} ...")
                self.update_idletasks()
                stack, orig_dtype, z_step_meta = load_microscopy_stack(ed.path)
                history = list(getattr(ed, "transform_history", []))
                if history:
                    self._log(
                        f"Background subtraction: applying {len(history)} geometry operation(s) for {base} ...")
                    self.update_idletasks()
                    stack = apply_stack_transform_history(stack, history)
                z_step = getattr(ed, "z_step_um", None) or z_step_meta
                if z_step is None or z_step <= 0:
                    if fallback_z is None:
                        fallback_z = self._ask_fallback_z_step()
                        if fallback_z is None:
                            self._log(f"Skipped {base}: no Z-step value available.")
                            skipped += 1
                            continue
                    z_step = fallback_z

                roi_rect = getattr(ed, "background_roi_rect", None)
                if roi_rect is not None:
                    mask = rectangle_mask_from_rect(roi_rect, stack.shape[1:])
                else:
                    mask = load_roi_mask_from_file(roi_path, stack.shape[1:])
                proj = project_stack_with_background_mask(
                    stack, mask, method="zdepth", z_step_um=z_step)

                out_name = build_filename(
                    base,
                    self.experiment_id,
                    ed.group,
                    kind="background_subtracted_Zdepth_normalised_sumIP",
                    ext=".tif",
                )
                out_path = os.path.join(out_dir, out_name)
                had_signal, _, _ = save_image16(out_path, proj)
                if not had_signal:
                    self._log(
                        f"⚠ {ed.name}: background-subtracted projection has no signal. Check the ROI."
                    )

                proj_ed = SampleEditor(
                    self.notebook,
                    out_path,
                    self,
                    image_array=proj,
                    name=out_name,
                    group=ed.group,
                )
                proj_ed.source_base = base
                proj_ed.normalized = proj
                self.editors.append(proj_ed)
                self.notebook.add(proj_ed, text=proj_ed.name[:24])
                self.notebook.select(proj_ed)
                created += 1
                self._log(f"Background-subtracted projection → {out_name}")
            except Exception as exc:
                skipped += 1
                self._log(f"Skipped {base}: background subtraction failed: {exc}")
            finally:
                # Drop large arrays before the next file. This is intentionally
                # explicit because CZI stacks can be hundreds of MB or larger.
                try:
                    del stack
                except Exception:
                    pass
                try:
                    del mask
                except Exception:
                    pass
                try:
                    del proj
                except Exception:
                    pass
                gc.collect()
                self.update_idletasks()

        self._refresh_file_bar()
        messagebox.showinfo(
            "Background subtraction done",
            f"Created {created} projection(s).\nSkipped {skipped}.\nOutput folder:\n{out_dir}",
        )

    def add_straighten_guides(self):
        """Add default straightening landmarks to the current 2-D projection."""
        ed = self._current_editor()
        if ed is None:
            self._log("No image loaded.")
            return
        if getattr(ed, "is_stack_editor", False):
            messagebox.showinfo(
                "Projection required",
                "Add straighten guides is available only on 2-D projection images. Use Tools → Projection first."
            )
            return
        if getattr(ed, "generated_preview", False):
            messagebox.showinfo(
                "Not applicable",
                "This is a straightened preview file. Add guides to the source projection instead."
            )
            return
        if getattr(ed, "has_straighten_guides", False):
            self._log(f"{ed.name}: straighten guides already exist.")
            return
        ed.initialize_straighten_guides()
        self._log(f"{ed.name}: added straighten guides.")

    def add_images(self):
        paths = filedialog.askopenfilenames(
            title="Open microscopy image, stack, or projection",
            filetypes=[
                ("Microscopy files", "*.czi *.ome.tif *.ome.tiff *.tif *.tiff *.png *.jpg *.jpeg *.lif *.nd2 *.lsm"),
                ("CZI files", "*.czi"),
                ("TIFF images", "*.tif *.tiff *.ome.tif *.ome.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        for p in paths:
            try:
                ext = os.path.splitext(p)[1].lower()

                # Large CZI files are registered lazily. Do not load every full
                # Z stack at Open time, because that can exhaust RAM and freeze
                # the GUI when many files are selected.
                if ext == ".czi":
                    ed = DeferredCziEditor(
                        self.notebook,
                        p,
                        self,
                        name=os.path.basename(p),
                        group=(self.conditions[0] if self.conditions else ""),
                    )
                    self.editors.append(ed)
                    self.notebook.add(ed, text=ed.name[:24])
                    roi_note = " + ROI" if ed.saved_roi_path else ""
                    self._log(f"Registered CZI {ed.name}{roi_note} (lazy-loaded).")
                    continue

                stack, orig_dtype, z_step_um = load_microscopy_stack(p)
                is_microscopy_container = ext in {".lif", ".nd2", ".lsm"}
                if stack.shape[0] > 1 or is_microscopy_container:
                    ed = StackEditor(self.notebook, p, self, stack,
                                     orig_dtype=orig_dtype,
                                     z_step_um=z_step_um,
                                     group=(self.conditions[0] if self.conditions else ""))
                    self.editors.append(ed)
                    self.notebook.add(ed, text=ed.name[:24])
                    z_note = f", Z-step {z_step_um:g} µm" if z_step_um else ""
                    self._log(f"Opened stack {ed.name}  (Z={stack.shape[0]}, {ed.w}x{ed.h}{z_note})")
                else:
                    ed = SampleEditor(self.notebook, p, self,
                                      image_array=stack[0],
                                      name=os.path.basename(p),
                                      group=(self.conditions[0] if self.conditions else ""))
                    self.editors.append(ed)
                    self.notebook.add(ed, text=ed.name[:24])
                    self._log(f"Opened image {ed.name}  ({ed.w}x{ed.h})")
            except Exception as exc:
                self._log(f"Skipped {os.path.basename(p)}: {exc}")
                continue
        if self.editors:
            self.notebook.select(self.editors[-1])
        self._refresh_file_bar()

    def _pick_condition(self, title="Select condition for this batch"):
        """Modal dialog to pick one of the experiment's conditions (with an
        option to add a new one). Returns the chosen condition string, or
        None if cancelled."""
        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)

        result = {"value": None}

        ttk.Label(dlg, text="Condition:").pack(anchor=tk.W, padx=12,
                                               pady=(12, 4))
        cond_var = tk.StringVar(value=(self.conditions[0]
                                       if self.conditions else "control"))
        combo = ttk.Combobox(dlg, textvariable=cond_var,
                             values=list(self.conditions), state="readonly",
                             width=30)
        combo.pack(fill=tk.X, padx=12)

        newframe = ttk.Frame(dlg)
        newframe.pack(fill=tk.X, padx=12, pady=(6, 0))
        ttk.Label(newframe, text="or add new:").pack(side=tk.LEFT)
        new_var = tk.StringVar(value="")
        ttk.Entry(newframe, textvariable=new_var, width=20).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        def confirm():
            new = new_var.get().strip()
            if new:
                if new not in self.conditions:
                    self.conditions.append(new)
                result["value"] = new
            else:
                result["value"] = cond_var.get().strip() or None
            dlg.destroy()

        btns = ttk.Frame(dlg)
        btns.pack(fill=tk.X, padx=12, pady=12)
        ttk.Button(btns, text="Cancel",
                   command=dlg.destroy).pack(side=tk.LEFT)
        ttk.Button(btns, text="OK", command=confirm).pack(side=tk.RIGHT)

        dlg.bind("<Return>", lambda e: confirm())
        self.wait_window(dlg)
        return result["value"]

    def import_czi(self):
        """Select CZI files, build normalized projections, save them into the
        experiment's normalized/ folder, and load them."""
        if not _HAVE_CZI:
            messagebox.showinfo(
                "CZI import unavailable",
                "CZI import needs aicsimageio (or bioio) + roifile + "
                "scikit-image.\nInstall them, or open existing 16-bit "
                "projection images with 'Open'.")
            return
        if not self.experiment_dir:
            messagebox.showwarning("No experiment",
                                   "No ExperimentID is set.")
            return

        condition = self._pick_condition()
        if condition is None:
            return

        paths = filedialog.askopenfilenames(
            title=f"Select CZI files for condition '{condition}' "
                  "(each needs a matching <name>.roi)",
            filetypes=[("CZI images", "*.czi"), ("All files", "*.*")],
        )
        if not paths:
            return

        norm_dir = os.path.join(self.experiment_dir, "normalized")
        try:
            os.makedirs(norm_dir, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Output folder error", str(exc))
            return

        n_loaded = 0
        for p in paths:
            base = os.path.splitext(os.path.basename(p))[0]
            roi_path = find_roi_for_czi(p)
            if roi_path is None:
                self._log(f"Skipped {base}.czi: no matching {base}.roi "
                          "in the same folder.")
                continue
            self._log(f"Processing {base}.czi …")
            self.update_idletasks()
            try:
                proj = czi_to_projection(p, roi_path)
            except Exception as exc:
                self._log(f"Skipped {base}.czi: {exc}")
                continue

            # auto-save the pure normalized projection into the experiment dir
            norm_name = build_filename(base, self.experiment_id, condition,
                                       kind="normalized")
            norm_path = os.path.join(norm_dir, norm_name)
            try:
                had_signal, vmin, vmax = save_image16(norm_path, proj)
                if not had_signal:
                    self._log(f"⚠ {base}: normalized projection has NO signal "
                              "(all zero) — check the ROI background region; "
                              "it may overlap the specimen.")
            except Exception as exc:
                self._log(f"{base}: could not save normalized image: {exc}")
                norm_path = p  # fall back to the czi path as identity

            try:
                ed = SampleEditor(self.notebook, norm_path, self,
                                  image_array=proj,
                                  name=norm_name, group=condition)
            except Exception as exc:
                self._log(f"Skipped {base}: could not open editor: {exc}")
                continue
            ed.source_base = base
            ed.normalized = proj
            self.editors.append(ed)
            self.notebook.add(ed, text=ed.name[:24])
            n_loaded += 1
            self._log(f"Saved normalized → {norm_name}  ({ed.w}x{ed.h})")

        if self.editors:
            self.notebook.select(self.editors[-1])
        self._refresh_file_bar()
        if n_loaded:
            self._log(f"CZI import done: {n_loaded} projection(s) for "
                      f"condition '{condition}'.")

    def _current_editor(self):
        if not self.editors:
            return None
        try:
            cur = self.notebook.select()
            for ed in self.editors:
                if str(ed) == str(cur):
                    return ed
        except Exception:
            pass
        return self.editors[-1]

    def remove_current(self):
        """Backward-compatible close action; the UI uses per-file X buttons."""
        ed = self._current_editor()
        if ed is not None:
            self.close_editor(ed)

    def straighten_current(self):
        ed = self._current_editor()
        if ed is None:
            self._log("No image loaded.")
            return
        if ed.generated_preview:
            self._log(f"{ed.name}: already a straightened preview file.")
            return
        if not self.experiment_dir:
            messagebox.showwarning("No experiment", "No ExperimentID is set.")
            return
        ok = ed.compute_straighten()
        if not ok:
            self._log(f"{ed.name}: FAILED to straighten")
            return

        str_dir = os.path.join(self.experiment_dir, "straightened")
        try:
            os.makedirs(str_dir, exist_ok=True)
        except Exception as exc:
            self._log(f"{ed.name}: could not create straightened/ : {exc}")
            return

        str_name = build_filename(ed.source_base, self.experiment_id, ed.group,
                                  kind="straightened")
        out_path = os.path.join(str_dir, str_name)

        # If a previous straightened file for this sample had a different name
        # (e.g. the condition changed), remove the stale file (and its preview).
        if ed.straightened_path and ed.straightened_path != out_path:
            _remove_image_and_preview(ed.straightened_path)

        # Close any previous preview tab for this sample before making a new one.
        if ed.preview_editor is not None and ed.preview_editor in self.editors:
            self.close_editor(ed.preview_editor)
            ed.preview_editor = None

        try:
            # overwrite any existing file with the same name
            had_signal, _, _ = save_image16(out_path, ed.straightened)
            if not had_signal:
                self._log(f"⚠ {ed.name}: straightened image has NO signal "
                          "(all zero) — check the source image and landmarks.")
            ed.straightened_path = out_path
            preview = SampleEditor(self.notebook, out_path, self,
                                   generated_preview=True)
            preview.straightened = preview.img.copy()
            ed.preview_editor = preview
            self.editors.append(preview)
            self.notebook.add(preview, text=preview.name[:24])
            self.notebook.select(preview)
            self._refresh_file_bar()
            self._log(f"{ed.name}: straightened → {str_name}")
        except Exception as exc:
            self._log(f"{ed.name}: straightened, but could not save/open: {exc}")

    # ------------------------------------------------------------------
    def _pick_heatmap_options(self):
        """
        Modal dialog: choose colormap (built-in or custom) and output formats
        (PDF and/or PNG). Returns dict {cmap, pdf, png} or None if cancelled.
        """
        dlg = tk.Toplevel(self)
        dlg.title("Heatmap options")
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)

        result = {"value": None}

        ttk.Label(dlg, text="Colour palette:").grid(
            row=0, column=0, sticky="w", padx=12, pady=(12, 2))

        builtin = ["viridis", "magma", "inferno", "plasma", "cividis",
                   "gray", "Custom…"]
        cmap_var = tk.StringVar(value="viridis")
        combo = ttk.Combobox(dlg, textvariable=cmap_var, values=builtin,
                             state="readonly", width=22)
        combo.grid(row=0, column=1, sticky="we", padx=12, pady=(12, 2))

        ttk.Label(dlg, text="Custom colours (comma-separated,\n"
                            "used only when 'Custom…' is selected):",
                  justify="left").grid(row=1, column=0, columnspan=2,
                                       sticky="w", padx=12, pady=(6, 2))
        custom_var = tk.StringVar(value="black, orange, white")
        custom_entry = ttk.Entry(dlg, textvariable=custom_var, width=40)
        custom_entry.grid(row=2, column=0, columnspan=2, sticky="we",
                          padx=12, pady=(0, 4))
        custom_entry.configure(state="disabled")

        def _on_cmap_change(*_a):
            if cmap_var.get() == "Custom…":
                custom_entry.configure(state="normal")
            else:
                custom_entry.configure(state="disabled")
        cmap_var.trace_add("write", _on_cmap_change)

        # format checkboxes
        fmt_frame = ttk.LabelFrame(dlg, text="Output files")
        fmt_frame.grid(row=3, column=0, columnspan=2, sticky="we",
                       padx=12, pady=(8, 4))
        pdf_var = tk.BooleanVar(value=True)
        png_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(fmt_frame, text="PDF (vector)",
                        variable=pdf_var).pack(side=tk.LEFT, padx=8, pady=4)
        ttk.Checkbutton(fmt_frame, text="PNG (raster)",
                        variable=png_var).pack(side=tk.LEFT, padx=8, pady=4)

        status = tk.StringVar(value="")
        ttk.Label(dlg, textvariable=status, foreground="#c0392b").grid(
            row=4, column=0, columnspan=2, sticky="w", padx=12)

        def confirm():
            if not pdf_var.get() and not png_var.get():
                status.set("Select at least one output format.")
                return
            sel = cmap_var.get()
            cmap = custom_var.get().strip() if sel == "Custom…" else sel
            if sel == "Custom…" and not cmap:
                status.set("Enter custom colours, or pick a built-in palette.")
                return
            result["value"] = {"cmap": cmap,
                               "pdf": pdf_var.get(),
                               "png": png_var.get()}
            dlg.destroy()

        btns = ttk.Frame(dlg)
        btns.grid(row=5, column=0, columnspan=2, sticky="we", padx=12,
                  pady=12)
        ttk.Button(btns, text="Cancel",
                   command=dlg.destroy).pack(side=tk.LEFT)
        ttk.Button(btns, text="Generate",
                   command=confirm).pack(side=tk.RIGHT)

        dlg.columnconfigure(1, weight=1)
        dlg.bind("<Return>", lambda e: confirm())
        self.wait_window(dlg)
        return result["value"]

    def _show_heatmap_window(self, fig, title="Cohort heatmap"):
        """Display a Matplotlib figure in a new resizable window."""
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("900x650")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        nav = NavigationToolbar2Tk(canvas, win, pack_toolbar=False)
        nav.update()
        nav.pack(side=tk.BOTTOM, fill=tk.X)

    def process_all(self):
        if not self.editors:
            messagebox.showinfo("Nothing to do", "Load some images first.")
            return
        if not self.experiment_dir:
            messagebox.showwarning("No experiment", "No ExperimentID is set.")
            return

        out_dir = self.experiment_dir
        str_dir = os.path.join(out_dir, "straightened")
        try:
            os.makedirs(str_dir, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Output folder error", str(exc))
            return

        profiles = []          # list of (name, group, trimmed)
        n_ok = 0
        for ed in self.editors:
            if ed.generated_preview:
                continue
            # straighten on demand if not done yet
            if ed.straightened is None:
                ed.compute_straighten()
            if ed.straightened is None:
                self._log(f"{ed.name}: skipped (could not straighten).")
                continue

            # make sure a straightened file exists on disk for this sample
            # (samples straightened interactively are already saved; this
            # covers any straightened on-demand just now).
            str_name = build_filename(ed.source_base, self.experiment_id,
                                      ed.group, kind="straightened")
            out_path = os.path.join(str_dir, str_name)
            if ed.straightened_path != out_path:
                if ed.straightened_path and ed.straightened_path != out_path:
                    _remove_image_and_preview(ed.straightened_path)
                try:
                    save_image16(out_path, ed.straightened)
                    ed.straightened_path = out_path
                except Exception as exc:
                    self._log(f"{ed.name}: could not save straightened image: {exc}")

            # full-length profile, no trimming
            sig = column_sums(ed.straightened)
            sig = np.asarray(sig, dtype=np.float64)
            if len(sig) == 0 or not np.any(np.isfinite(sig)):
                self._log(f"{ed.name}: empty profile — skipped.")
                continue

            profiles.append((ed.name, ed.group, sig))
            n_ok += 1
            grp = ed.group if ed.group else "(no condition)"
            self._log(f"{ed.name}: full profile length {len(sig)} "
                      f"(condition: {grp}; no trimming).")

        if not profiles:
            messagebox.showwarning("No profiles",
                                   "No usable profiles were produced.")
            return

        names, groups, matrix = assemble_matrix(profiles)

        # ask for palette and output formats
        opts = self._pick_heatmap_options()
        if opts is None:
            self._log("Heatmap cancelled; tables will still be written.")
            n_groups = 0
        else:
            try:
                fig, n_groups = self._build_cohort_heatmap_figure(
                    names, groups, matrix, cmap_spec=opts["cmap"])
                base = build_filename("COHORT_HEATMAP", self.experiment_id, "",
                                      ext="")  # no extension yet
                if opts["pdf"]:
                    p = os.path.join(out_dir, base + ".pdf")
                    fig.savefig(p, format="pdf", bbox_inches="tight")
                    self._log(f"Wrote {p}")
                if opts["png"]:
                    p = os.path.join(out_dir, base + ".png")
                    fig.savefig(p, format="png", dpi=200, bbox_inches="tight")
                    self._log(f"Wrote {p}")
                # display in a new window
                self._show_heatmap_window(
                    fig, title=f"Cohort heatmap — {self.experiment_id}")
            except Exception as exc:
                n_groups = 0
                self._log(f"Cohort heatmap failed: {exc}")

        # tables, named by ExperimentID
        try:
            written = write_tables(out_dir, self.experiment_id,
                                   names, groups, matrix)
            for w in written:
                self._log(f"Wrote {w}")
        except Exception as exc:
            self._log(f"Table export failed: {exc}")

        cond_note = (f"\n{n_groups} conditions color-coded on the heatmap."
                     if n_groups and n_groups > 1 else "")
        messagebox.showinfo(
            "Done",
            f"Processed {n_ok}/{len(self.editors)} samples.\n"
            f"Output in:\n{out_dir}{cond_note}")
        self._log(f"FINISHED: {n_ok}/{len(self.editors)} samples → {out_dir}")

    # ------------------------------------------------------------------
    def _resolve_colormap(self, cmap_spec):
        """
        Resolve a colormap specification to a Matplotlib colormap object.

        cmap_spec may be:
          - a built-in colormap name (e.g. "viridis", "magma")
          - a comma/space-separated list of colors for a custom continuous map
            (e.g. "black, orange, white" or "#000000,#ff8800,#ffffff")
        Falls back to viridis on any failure.
        """
        from matplotlib.colors import LinearSegmentedColormap
        spec = (cmap_spec or "viridis").strip()

        # custom color list -> continuous colormap
        if any(sep in spec for sep in (",",)) or " " in spec.strip():
            tokens = [t.strip() for t in re.split(r"[,\s]+", spec) if t.strip()]
            # only treat as a color list if it is not just a single name
            if len(tokens) >= 2:
                try:
                    return LinearSegmentedColormap.from_list("custom", tokens)
                except Exception:
                    pass

        try:
            return matplotlib.colormaps[spec].copy()
        except Exception:
            try:
                return cm.get_cmap(spec).copy()
            except Exception:
                return matplotlib.colormaps["viridis"].copy()

    def _build_cohort_heatmap_figure(self, names, groups, matrix,
                                     cmap_spec="viridis"):
        """
        Build and return (figure, n_distinct_conditions) for the cohort heatmap.
        When more than one distinct condition is present, samples are sorted by
        condition and a colored sidebar strip plus legend encode condition.
        The figure height scales with the number of samples.
        """
        names = list(names)
        groups = list(groups)
        matrix = np.asarray(matrix, dtype=np.float64)

        distinct = [g for g in dict.fromkeys(groups) if g]  # ordered, non-empty
        multi = len(distinct) > 1

        if multi:
            order = sorted(range(len(names)),
                           key=lambda i: (groups[i] == "",  # blanks last
                                          groups[i], names[i]))
            names = [names[i] for i in order]
            groups = [groups[i] for i in order]
            matrix = matrix[order]

        masked = np.ma.masked_invalid(matrix)

        base_cmap = self._resolve_colormap(cmap_spec)
        base_cmap.set_bad(color="lightgrey")

        # --- proportional sizing ---
        n_rows = max(1, len(names))
        row_h = 0.32                      # inches per sample row
        fig_h = min(max(2.2, 1.4 + row_h * n_rows), 60.0)
        n_cols = matrix.shape[1] if matrix.ndim == 2 and matrix.size else 1
        fig_w = min(max(7.0, 4.0 + n_cols / 110.0), 20.0)
        fig = Figure(figsize=(fig_w, fig_h), dpi=200)

        if multi:
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 40], wspace=0.02)
            ax_strip = fig.add_subplot(gs[0, 0])
            ax = fig.add_subplot(gs[0, 1])

            all_conditions = distinct + (
                [""] if any(g == "" for g in groups) else [])
            try:
                qual = matplotlib.colormaps["tab10"]
            except Exception:
                qual = cm.get_cmap("tab10")
            color_map = {}
            for k, cond in enumerate(distinct):
                color_map[cond] = qual(k % 10)
            color_map[""] = (0.8, 0.8, 0.8, 1.0)  # grey for unlabeled

            strip = np.array([[ {c: i for i, c in enumerate(all_conditions)}[g]
                                for g in groups ]]).T
            from matplotlib.colors import ListedColormap
            strip_cmap = ListedColormap(
                [color_map[c] for c in all_conditions])
            ax_strip.imshow(strip, aspect="auto", cmap=strip_cmap,
                            interpolation="nearest")
            ax_strip.set_xticks([])
            ax_strip.set_yticks(range(len(names)))
            ax_strip.set_yticklabels([n[:28] for n in names], fontsize=7)
            ax_strip.set_title("cond.", fontsize=8)

            im = ax.imshow(masked, aspect="auto", cmap=base_cmap,
                           interpolation="nearest")
            ax.set_yticks([])
            ax.set_xlabel("Position along spine (trimmed pixels)")
            ax.set_title("Cohort mineralization heatmap")
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01,
                         label="Integrated intensity")

            from matplotlib.patches import Patch
            handles = [Patch(facecolor=color_map[c], label=c)
                       for c in distinct]
            if any(g == "" for g in groups):
                handles.append(Patch(facecolor=color_map[""],
                                     label="(no condition)"))
            ax.legend(handles=handles, title="Condition",
                      bbox_to_anchor=(1.12, 1.0), loc="upper left",
                      fontsize=7, title_fontsize=8, frameon=False)
        else:
            ax = fig.add_subplot(111)
            im = ax.imshow(masked, aspect="auto", cmap=base_cmap,
                           interpolation="nearest")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([n[:28] for n in names], fontsize=7)
            ax.set_xlabel("Position along spine (trimmed pixels)")
            ax.set_title("Cohort mineralization heatmap")
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01,
                         label="Integrated intensity")
            fig.tight_layout()

        return fig, len(distinct)

    def _save_cohort_heatmap(self, hm_path, names, groups, matrix,
                             cmap_spec="viridis"):
        """
        Build the cohort heatmap and save it to hm_path. The output format is
        taken from the file extension (.pdf or .png). Returns the number of
        distinct non-empty conditions. (Kept for backward compatibility.)
        """
        fig, n_distinct = self._build_cohort_heatmap_figure(
            names, groups, matrix, cmap_spec)
        ext = os.path.splitext(hm_path)[1].lower().lstrip(".") or "pdf"
        fig.savefig(hm_path, format=ext, bbox_inches="tight")
        return n_distinct




# ======================================================================
#  Runtime patches: smart Open and stack-aware sidebar / processing
# ======================================================================
def _scamp_open_smart(self):
    """Open common microscopy files. Stacks open in StackEditor; 2-D images open in SampleEditor."""
    paths = filedialog.askopenfilenames(
        title="Open microscopy image, hyperstack, or 16-bit projection",
        filetypes=[
            ("Microscopy / image files", "*.czi *.lif *.nd2 *.lsm *.ome.tif *.ome.tiff *.tif *.tiff *.png *.jpg *.jpeg"),
            ("All files", "*.*"),
        ],
    )
    if not paths:
        return
    default_group = self.conditions[0] if getattr(self, "conditions", None) else ""
    for p in paths:
        try:
            stack, orig_dtype, z_step_um = load_microscopy_stack(p)
            if stack.shape[0] > 1:
                ed = StackEditor(self.notebook, p, self, stack, orig_dtype=orig_dtype,
                                 z_step_um=z_step_um, group=default_group)
                self._log(f"Opened stack {ed.name}  (Z={stack.shape[0]}, {ed.w}x{ed.h})")
            else:
                ed = SampleEditor(self.notebook, p, self,
                                  image_array=stack[0], name=os.path.basename(p),
                                  group=default_group)
                self._log(f"Opened image {ed.name}  ({ed.w}x{ed.h})")
        except Exception as exc:
            self._log(f"Skipped {os.path.basename(p)}: {exc}")
            continue
        self.editors.append(ed)
        self.notebook.add(ed, text=ed.name[:24])
    if self.editors:
        self.notebook.select(self.editors[-1])
    self._refresh_file_bar()


def _scamp_refresh_file_bar_stack_aware(self):
    """Rebuild the sidebar and support both image editors and stack editors."""
    for child in self.file_list.winfo_children():
        child.destroy()
    for ed in self.editors:
        row = ttk.Frame(self.file_list)
        row.pack(side=tk.TOP, fill=tk.X, padx=4, pady=1)
        top_line = ttk.Frame(row)
        top_line.pack(side=tk.TOP, fill=tk.X)
        label = ed.name[:24] + ("..." if len(ed.name) > 24 else "")
        if getattr(ed, "generated_preview", False):
            label = "↳ " + label
        if getattr(ed, "is_stack_editor", False):
            label = "▣ " + label
        ttk.Button(top_line, text=label,
                   command=lambda e=ed: self._select_editor(e)).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(top_line, text="x", width=2, fg="#c0392b",
                  activeforeground="#e74c3c", font=("TkDefaultFont", 10, "bold"),
                  relief=tk.FLAT, bd=0, padx=4, cursor="hand2",
                  command=lambda e=ed: self.close_editor(e)).pack(side=tk.RIGHT, padx=(2, 0))

        roi_path = getattr(ed, "saved_roi_path", None)
        if roi_path:
            roi_line = ttk.Frame(row)
            roi_line.pack(side=tk.TOP, fill=tk.X, pady=(1, 1))
            roi_label = "ROI: " + os.path.basename(roi_path)
            ttk.Label(roi_line, text=roi_label, foreground="#555").pack(side=tk.LEFT, padx=(8, 0))

        if not getattr(ed, "generated_preview", False):
            cond_line = ttk.Frame(row)
            cond_line.pack(side=tk.TOP, fill=tk.X, pady=(1, 3))
            ttk.Label(cond_line, text="cond:").pack(side=tk.LEFT)
            choices = list(getattr(self, "conditions", []) or [])
            if getattr(ed, "group", "") and ed.group not in choices:
                choices.append(ed.group)
            var = tk.StringVar(value=getattr(ed, "group", "") or (choices[0] if choices else ""))
            combo = ttk.Combobox(cond_line, textvariable=var, values=choices,
                                 state="readonly", width=14)
            combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
            def _on_change(*_a, e=ed, v=var):
                e.group = v.get().strip()
            var.trace_add("write", _on_change)


def _scamp_straighten_current_stack_guard(self):
    ed = self._current_editor()
    if ed is None:
        self._log("No image loaded.")
        return
    if getattr(ed, "is_stack_editor", False):
        messagebox.showinfo(
            "Projection required",
            "This is a stack. Use Tools → Projection first, then straighten the generated 16-bit projection."
        )
        return
    return _SCAMP_ORIGINAL_STRAIGHTEN_CURRENT(self)


def _scamp_process_all_stack_guard(self):
    # The original function already skips generated previews. Stack editors do
    # not have straightening landmarks, so skip them as raw inputs.
    raw_editors = list(self.editors)
    stack_editors = [ed for ed in raw_editors if getattr(ed, "is_stack_editor", False)]
    if stack_editors:
        self._log(f"Skipping {len(stack_editors)} raw stack(s). Create projections before batch processing.")
    self.editors = [ed for ed in raw_editors if not getattr(ed, "is_stack_editor", False)]
    try:
        return _SCAMP_ORIGINAL_PROCESS_ALL(self)
    finally:
        self.editors = raw_editors


# Apply the patches after App exists.
_SCAMP_ORIGINAL_STRAIGHTEN_CURRENT = App.straighten_current
_SCAMP_ORIGINAL_PROCESS_ALL = App.process_all
App.add_images = _scamp_open_smart
App._refresh_file_bar = _scamp_refresh_file_bar_stack_aware
App.straighten_current = _scamp_straighten_current_stack_guard
App.process_all = _scamp_process_all_stack_guard



# ======================================================================
#  Stability patch: true lazy CZI Open + memory-light projections
# ======================================================================
def _as_zyx_stack(arr):
    """Convert a loaded microscopy array to a grayscale ZYX stack using float32.

    The previous stack conversion used float64, which doubled memory pressure
    for large CZI files. float32 is enough for preview, background correction,
    and projection while keeping peak RAM much lower.
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        return a[np.newaxis, :, :].astype(np.float32, copy=False)
    if a.ndim == 3:
        if a.shape[-1] in (3, 4) and a.shape[0] > 4:
            rgb = a[..., :3]
            return rgb.mean(axis=-1)[np.newaxis, :, :].astype(np.float32, copy=False)
        return a.astype(np.float32, copy=False)

    a = np.squeeze(a)
    if a.ndim == 2:
        return a[np.newaxis, :, :].astype(np.float32, copy=False)
    if a.ndim == 3:
        return _as_zyx_stack(a)

    spatial = a.shape[-2:]
    lead_shape = a.shape[:-2]
    z_axis = None
    for i in reversed(range(len(lead_shape))):
        if lead_shape[i] > 1:
            z_axis = i
            break
    if z_axis is None:
        return a.reshape((-1,) + spatial).mean(axis=0)[np.newaxis, :, :].astype(np.float32, copy=False)

    a = np.moveaxis(a, z_axis, 0)
    z = a.shape[0]
    rest = int(np.prod(a.shape[1:-2]))
    a = a.reshape((z, rest) + spatial)
    if rest > 1:
        a = np.max(a, axis=1)
    else:
        a = a[:, 0]
    return a.astype(np.float32, copy=False)


def project_stack_with_background_mask(stack, mask, method="zdepth", z_step_um=None):
    """Memory-light background-corrected projection from a ZYX stack.

    This version avoids making a full corrected copy of the stack. It iterates
    over Z slices, subtracts the ROI background, clips negatives, and accumulates
    directly into one 2-D float32 image.
    """
    data = np.asarray(stack, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("Expected a ZYX stack.")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != data.shape[1:]:
        raise ValueError(
            f"ROI mask shape {mask.shape} does not match image shape {data.shape[1:]}."
        )
    if not mask.any():
        raise ValueError("ROI mask is empty.")

    method = method.lower()
    if method not in ("mean", "sum", "zdepth", "z-depth", "zdepth normalised sum"):
        raise ValueError(f"Unknown projection method: {method}")

    acc = np.zeros(data.shape[1:], dtype=np.float32)
    for z in range(data.shape[0]):
        sl = data[z].astype(np.float32, copy=False)
        bg = float(np.mean(sl[mask], dtype=np.float64))
        corr = sl - bg
        np.maximum(corr, 0, out=corr)
        acc += corr

    if method == "mean":
        return acc / float(data.shape[0])
    if method in ("zdepth", "z-depth", "zdepth normalised sum"):
        if z_step_um is None or z_step_um <= 0:
            raise ValueError("Z-depth normalised sum needs a positive Z-step in µm.")
        depth_um = float(data.shape[0]) * float(z_step_um)
        return acc / depth_um
    return acc


def _scamp_open_smart_lazy(self):
    """Open files with true lazy CZI registration.

    CZI files are not decoded at Open time. They appear in Open files with a
    visible Load preview button. Non-CZI stacks/images still open normally.
    """
    paths = filedialog.askopenfilenames(
        title="Open microscopy image, hyperstack, or 16-bit projection",
        filetypes=[
            ("Microscopy / image files", "*.czi *.lif *.nd2 *.lsm *.ome.tif *.ome.tiff *.tif *.tiff *.png *.jpg *.jpeg"),
            ("CZI files", "*.czi"),
            ("All files", "*.*"),
        ],
    )
    if not paths:
        return

    default_group = self.conditions[0] if getattr(self, "conditions", None) else ""
    opened = 0
    for p in paths:
        try:
            ext = os.path.splitext(p)[1].lower()
            if ext == ".czi":
                ed = DeferredCziEditor(
                    self.notebook,
                    p,
                    self,
                    name=os.path.basename(p),
                    group=default_group,
                )
                self._log(f"Registered CZI {ed.name} (not loaded into RAM).")
            else:
                stack, orig_dtype, z_step_um = load_microscopy_stack(p)
                if stack.shape[0] > 1 or ext in {".lif", ".nd2", ".lsm"}:
                    ed = StackEditor(
                        self.notebook, p, self, stack,
                        orig_dtype=orig_dtype, z_step_um=z_step_um,
                        group=default_group,
                    )
                    self._log(f"Opened stack {ed.name}  (Z={stack.shape[0]}, {ed.w}x{ed.h})")
                else:
                    ed = SampleEditor(
                        self.notebook, p, self,
                        image_array=stack[0], name=os.path.basename(p),
                        group=default_group,
                    )
                    self._log(f"Opened image {ed.name}  ({ed.w}x{ed.h})")
            self.editors.append(ed)
            self.notebook.add(ed, text=ed.name[:24])
            opened += 1
        except Exception as exc:
            self._log(f"Skipped {os.path.basename(p)}: {exc}")
            continue
        finally:
            gc.collect()
            self.update_idletasks()

    if opened:
        self.notebook.select(self.editors[-1])
    self._refresh_file_bar()


def _scamp_refresh_file_bar_stack_aware_lazy(self):
    """Sidebar with an explicit Load preview button for deferred CZI files."""
    for child in self.file_list.winfo_children():
        child.destroy()
    for ed in self.editors:
        row = ttk.Frame(self.file_list)
        row.pack(side=tk.TOP, fill=tk.X, padx=4, pady=1)

        top_line = ttk.Frame(row)
        top_line.pack(side=tk.TOP, fill=tk.X)
        label = ed.name[:24] + ("..." if len(ed.name) > 24 else "")
        if getattr(ed, "generated_preview", False):
            label = "↳ " + label
        elif getattr(ed, "is_deferred_czi", False):
            label = "CZI ⏸ " + label
        elif getattr(ed, "is_stack_editor", False):
            label = "Stack ▣ " + label
        ttk.Button(top_line, text=label,
                   command=lambda e=ed: self._select_editor(e)).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(top_line, text="x", width=2, fg="#c0392b",
                  activeforeground="#e74c3c", font=("TkDefaultFont", 10, "bold"),
                  relief=tk.FLAT, bd=0, padx=4, cursor="hand2",
                  command=lambda e=ed: self.close_editor(e)).pack(side=tk.RIGHT, padx=(2, 0))

        if getattr(ed, "is_deferred_czi", False):
            action_line = ttk.Frame(row)
            action_line.pack(side=tk.TOP, fill=tk.X, pady=(1, 1))
            ttk.Button(
                action_line,
                text="Load preview / edit ROI",
                command=lambda e=ed: e.load_full_stack_editor(),
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        roi_path = getattr(ed, "saved_roi_path", None)
        if roi_path and os.path.isfile(roi_path):
            roi_line = ttk.Frame(row)
            roi_line.pack(side=tk.TOP, fill=tk.X, pady=(1, 1))
            ttk.Label(roi_line, text="ROI: " + os.path.basename(roi_path),
                      foreground="#555").pack(side=tk.LEFT, padx=(8, 0))

        if not getattr(ed, "generated_preview", False):
            cond_line = ttk.Frame(row)
            cond_line.pack(side=tk.TOP, fill=tk.X, pady=(1, 3))
            ttk.Label(cond_line, text="cond:").pack(side=tk.LEFT)
            choices = list(getattr(self, "conditions", []) or [])
            if getattr(ed, "group", "") and ed.group not in choices:
                choices.append(ed.group)
            var = tk.StringVar(value=getattr(ed, "group", "") or (choices[0] if choices else ""))
            combo = ttk.Combobox(cond_line, textvariable=var, values=choices,
                                 state="readonly", width=14)
            combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
            def _on_change(*_a, e=ed, v=var):
                e.group = v.get().strip()
            var.trace_add("write", _on_change)

    # Update the scroll region after rebuilding the sidebar. The scrollbar is
    # shown only when the content is taller than the visible list area.
    try:
        self.update_idletasks()
        self.file_list_canvas.configure(scrollregion=self.file_list_canvas.bbox("all"))
        needed = (self.file_list.winfo_reqheight() > self.file_list_canvas.winfo_height() or len(getattr(self, "editors", [])) > 6)
        if needed and not self.file_list_scrollbar.winfo_ismapped():
            self.file_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, before=self.file_list_canvas)
        elif not needed and self.file_list_scrollbar.winfo_ismapped():
            self.file_list_scrollbar.pack_forget()
            self.file_list_canvas.yview_moveto(0)
    except Exception:
        pass


# Override the earlier smart-open patch. The earlier patch accidentally loaded
# CZI files immediately; this one keeps CZI files truly lazy and adds a visible
# sidebar button for loading one preview on demand.
App.add_images = _scamp_open_smart_lazy
App._refresh_file_bar = _scamp_refresh_file_bar_stack_aware_lazy

# ----------------------------------------------------------------------
# Robust sidebar scroll fix
# ----------------------------------------------------------------------
def _scamp_update_file_list_scrollbar(self):
    """Show the Open files scrollbar when the sidebar content overflows.

    The earlier implementation sometimes checked the canvas height before Tk
    had finished layout, so the scrollbar could stay hidden even with many
    files. This version runs after idle, uses both measured overflow and a
    conservative item-count fallback, and packs the scrollbar before the canvas
    so it always gets visible space.
    """
    try:
        self.update_idletasks()
        bbox = self.file_list_canvas.bbox("all")
        self.file_list_canvas.configure(scrollregion=bbox)
        content_h = (bbox[3] - bbox[1]) if bbox else 0
        canvas_h = max(1, self.file_list_canvas.winfo_height())
        needed = content_h > canvas_h or len(getattr(self, "editors", [])) > 6
        if needed:
            if not self.file_list_scrollbar.winfo_ismapped():
                self.file_list_scrollbar.pack(
                    side=tk.RIGHT, fill=tk.Y, before=self.file_list_canvas
                )
        else:
            if self.file_list_scrollbar.winfo_ismapped():
                self.file_list_scrollbar.pack_forget()
                self.file_list_canvas.yview_moveto(0)
    except Exception:
        pass


def _scamp_bind_sidebar_mousewheel(self):
    """Bind mouse-wheel scrolling while the pointer is over the file list."""
    def _wheel(event):
        try:
            if not self.file_list_scrollbar.winfo_ismapped():
                return
            if getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            else:
                delta = int(-1 * (event.delta / 120))
            self.file_list_canvas.yview_scroll(delta, "units")
        except Exception:
            pass

    def _bind(_event=None):
        try:
            self.bind_all("<MouseWheel>", _wheel)
            self.bind_all("<Button-4>", _wheel)
            self.bind_all("<Button-5>", _wheel)
        except Exception:
            pass

    def _unbind(_event=None):
        try:
            self.unbind_all("<MouseWheel>")
            self.unbind_all("<Button-4>")
            self.unbind_all("<Button-5>")
        except Exception:
            pass

    try:
        self.file_list_container.bind("<Enter>", _bind)
        self.file_list_container.bind("<Leave>", _unbind)
        self.file_list_canvas.bind("<Enter>", _bind)
        self.file_list_canvas.bind("<Leave>", _unbind)
        self.file_list.bind("<Enter>", _bind)
        self.file_list.bind("<Leave>", _unbind)
    except Exception:
        pass


# Wrap the active sidebar refresh function so the scrollbar update happens
# after Tk has finished creating and sizing all rows.
_SCAMP_PREVIOUS_REFRESH_FILE_BAR = App._refresh_file_bar

def _scamp_refresh_file_bar_with_reliable_scroll(self):
    _SCAMP_PREVIOUS_REFRESH_FILE_BAR(self)
    try:
        _scamp_bind_sidebar_mousewheel(self)
        self.after_idle(lambda: _scamp_update_file_list_scrollbar(self))
        self.after(100, lambda: _scamp_update_file_list_scrollbar(self))
    except Exception:
        pass

App._refresh_file_bar = _scamp_refresh_file_bar_with_reliable_scroll

# ----------------------------------------------------------------------
# Robust Process all fix
# ----------------------------------------------------------------------
def _scamp_group_from_straightened_name(filename, experiment_id):
    """Best-effort group parser for files named *_<ExperimentID>_straightened_<group>.tif."""
    root = os.path.splitext(os.path.basename(filename))[0]
    marker = f"_{experiment_id}_straightened_"
    if marker in root:
        return root.split(marker, 1)[1]
    marker = "_straightened_"
    if marker in root:
        return root.split(marker, 1)[1]
    return ""


def _scamp_profile_from_image_array(img_float):
    """Return the full-length 1-D column-sum profile from a 2-D image array.

    Process all intentionally does not trim the profile. Any zero-valued
    leading/trailing regions present in the straightened output are retained so
    that the exported matrix reflects the complete straightened image width.
    """
    sig = column_sums(img_float)
    sig = np.asarray(sig, dtype=np.float64)
    return sig, 0


def _scamp_process_all_robust(self):
    """Process profiles from memory and from saved straightened TIFFs.

    Older SCAMP logic only processed non-preview source editors and skipped
    generated straightened preview tabs. With lazy CZI/closeable previews, the
    source projection editor may no longer be the only reliable owner of the
    straightened data, while the straightened TIFFs are already written on disk.
    This version accepts:
      1. in-memory source editors with ed.straightened,
      2. generated straightened preview editors,
      3. saved *.tif files inside <ExperimentID>/straightened/.
    """
    if not getattr(self, "editors", None):
        messagebox.showinfo("Nothing to do", "Load some images first.")
        return
    if not self.experiment_dir:
        messagebox.showwarning("No experiment", "No ExperimentID is set.")
        return

    out_dir = self.experiment_dir
    str_dir = os.path.join(out_dir, "straightened")
    os.makedirs(str_dir, exist_ok=True)

    profiles = []
    seen = set()

    def add_profile(name, group, img, origin):
        key = os.path.abspath(origin) if origin else name
        if key in seen:
            return False
        try:
            profile, start = _scamp_profile_from_image_array(img)
        except Exception as exc:
            self._log(f"{name}: profile extraction failed: {exc}")
            return False
        if len(profile) == 0 or not np.any(np.isfinite(profile)):
            self._log(f"{name}: empty profile — skipped.")
            return False
        profiles.append((name, group or "", profile))
        seen.add(key)
        grp = group if group else "(no condition)"
        self._log(f"{name}: full profile length {len(profile)} (condition: {grp}; no trimming).")
        return True

    # 1) In-memory editors.
    for ed in list(self.editors):
        if getattr(ed, "is_stack_editor", False) or getattr(ed, "is_deferred_czi", False):
            continue

        if getattr(ed, "generated_preview", False):
            # Generated preview tabs are straightened files. The old code skipped
            # these, which could leave Process all with zero profiles.
            img = getattr(ed, "img", None)
            if img is not None:
                add_profile(ed.name, getattr(ed, "group", ""), img, getattr(ed, "path", ed.name))
            continue

        # Normal projection source editor: use existing straightened data, or
        # attempt to straighten on demand if guides exist.
        if getattr(ed, "straightened", None) is None:
            try:
                ed.compute_straighten()
            except Exception as exc:
                self._log(f"{ed.name}: skipped (could not straighten: {exc}).")
                continue
        if getattr(ed, "straightened", None) is None:
            self._log(f"{ed.name}: skipped (could not straighten).")
            continue

        str_name = build_filename(ed.source_base, self.experiment_id, getattr(ed, "group", ""), kind="straightened")
        out_path = os.path.join(str_dir, str_name)
        if getattr(ed, "straightened_path", None) != out_path:
            try:
                save_image16(out_path, ed.straightened)
                ed.straightened_path = out_path
            except Exception as exc:
                self._log(f"{ed.name}: could not save straightened image: {exc}")
        add_profile(ed.name, getattr(ed, "group", ""), ed.straightened, out_path)

    # 2) Disk fallback: scan saved straightened TIFFs. This is the critical fix
    # for closeable/lazy workflows where the file exists but the source editor
    # is no longer the active owner of the image array.
    try:
        disk_files = []
        for fn in os.listdir(str_dir):
            low = fn.lower()
            if low.endswith((".tif", ".tiff")) and "_straightened" in low and "_preview" not in low:
                disk_files.append(os.path.join(str_dir, fn))
        for path in sorted(disk_files):
            key = os.path.abspath(path)
            if key in seen:
                continue
            try:
                img, _dtype = load_image_any(path)
            except Exception as exc:
                self._log(f"{os.path.basename(path)}: could not read straightened TIFF: {exc}")
                continue
            group = _scamp_group_from_straightened_name(path, self.experiment_id)
            add_profile(os.path.basename(path), group, img, path)
    except Exception as exc:
        self._log(f"Could not scan straightened folder: {exc}")

    if not profiles:
        messagebox.showwarning(
            "No profiles",
            "No usable profiles were produced. Check that straightened TIFFs are readable."
        )
        self._log("No usable profiles were produced after checking memory and the straightened/ folder.")
        return

    names, groups, matrix = assemble_matrix(profiles)

    opts = self._pick_heatmap_options()
    n_groups = 0
    if opts is None:
        self._log("Heatmap cancelled; tables will still be written.")
    else:
        try:
            fig, n_groups = self._build_cohort_heatmap_figure(names, groups, matrix, cmap_spec=opts["cmap"])
            base = build_filename("COHORT_HEATMAP", self.experiment_id, "", ext="")
            if opts.get("pdf"):
                out_pdf = os.path.join(out_dir, base + ".pdf")
                fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
                self._log(f"Wrote {out_pdf}")
            if opts.get("png"):
                out_png = os.path.join(out_dir, base + ".png")
                fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
                self._log(f"Wrote {out_png}")
            self._show_heatmap_window(fig, title=f"Cohort heatmap — {self.experiment_id}")
        except Exception as exc:
            self._log(f"Cohort heatmap failed: {exc}")

    try:
        written = write_tables(out_dir, self.experiment_id, names, groups, matrix)
        for w in written:
            self._log(f"Wrote {w}")
    except Exception as exc:
        self._log(f"Table export failed: {exc}")

    cond_note = (f"\n{n_groups} conditions color-coded on the heatmap." if n_groups and n_groups > 1 else "")
    messagebox.showinfo(
        "Done",
        f"Processed {len(profiles)} profile(s).\nOutput in:\n{out_dir}{cond_note}"
    )
    self._log(f"FINISHED: {len(profiles)} profile(s) → {out_dir}")


# Install the robust Process all handler last, after all earlier stack guards.
App.process_all = _scamp_process_all_robust

# ======================================================================
#  Entry point
# ======================================================================
def main():
    try:
        app = App()
    except Exception:
        sys.stderr.write("Failed to start GUI:\n")
        traceback.print_exc()
        sys.exit(1)

    def _excepthook(exc_type, exc_value, exc_tb):
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        try:
            app._log("UNEXPECTED ERROR:\n" + msg)
            messagebox.showerror("Unexpected error", str(exc_value))
        except Exception:
            sys.stderr.write(msg)

    sys.excepthook = _excepthook
    app.mainloop()


# ======================================================================
#  QC + configurable background estimator patch
# ======================================================================
def _scamp_qc_dir(app):
    """Return the experiment QC folder and create it when possible."""
    base = getattr(app, "experiment_dir", None) or os.getcwd()
    path = os.path.join(base, "qc_reports")
    os.makedirs(path, exist_ok=True)
    return path


def _scamp_write_qc_report(app, base_name, step_name, qc):
    """Write one QC report as JSON and one-row CSV. Returns (json, csv)."""
    qc_dir = _scamp_qc_dir(app)
    safe_base = _sanitize(base_name)
    safe_step = _sanitize(step_name)
    json_path = os.path.join(qc_dir, f"{safe_base}_{safe_step}_qc.json")
    csv_path = os.path.join(qc_dir, f"{safe_base}_{safe_step}_qc.csv")
    payload = dict(qc)
    payload["qc_json"] = json_path
    payload["qc_csv"] = csv_path
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)
    return json_path, csv_path


def _scamp_float(x):
    """JSON-safe finite float, or None."""
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def _scamp_percent_change(before, after):
    before = float(before)
    after = float(after)
    if not math.isfinite(before) or abs(before) < 1e-12:
        return None
    return (after - before) / before * 100.0


def _scamp_estimate_background(values, estimator="median", percentile=20.0):
    """Estimate background from ROI pixels using a robust selectable estimator."""
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    estimator = str(estimator or "median").lower()
    if estimator == "mean":
        return float(np.mean(arr, dtype=np.float64))
    if estimator == "median":
        return float(np.median(arr))
    if estimator == "percentile":
        p = max(0.0, min(100.0, float(percentile)))
        return float(np.percentile(arr, p))
    raise ValueError(f"Unknown background estimator: {estimator}")


def project_stack_with_background_mask_qc(stack, mask, method="zdepth", z_step_um=None,
                                          estimator="median", percentile=20.0):
    """Memory-light background-corrected projection plus QC statistics."""
    data = np.asarray(stack, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("Expected a ZYX stack.")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != data.shape[1:]:
        raise ValueError(
            f"ROI mask shape {mask.shape} does not match image shape {data.shape[1:]}.")
    if not mask.any():
        raise ValueError("ROI mask is empty.")

    method_norm = str(method).lower()
    if method_norm not in ("mean", "sum", "zdepth", "z-depth", "zdepth normalised sum"):
        raise ValueError(f"Unknown projection method: {method}")

    acc = np.zeros(data.shape[1:], dtype=np.float32)
    bg_values = []
    raw_total = 0.0
    corrected_total = 0.0
    clipped_pixels = 0
    total_pixels = int(data.size)

    for z in range(data.shape[0]):
        sl = data[z].astype(np.float32, copy=False)
        bg = _scamp_estimate_background(sl[mask], estimator=estimator, percentile=percentile)
        bg_values.append(bg)
        raw_total += float(np.sum(sl, dtype=np.float64))
        corr = sl - bg
        clipped_pixels += int(np.count_nonzero(corr < 0))
        np.maximum(corr, 0, out=corr)
        corrected_total += float(np.sum(corr, dtype=np.float64))
        acc += corr

    if method_norm == "mean":
        proj = acc / float(data.shape[0])
    elif method_norm in ("zdepth", "z-depth", "zdepth normalised sum"):
        if z_step_um is None or z_step_um <= 0:
            raise ValueError("Z-depth normalised sum needs a positive Z-step in µm.")
        depth_um = float(data.shape[0]) * float(z_step_um)
        proj = acc / depth_um
    else:
        proj = acc

    bg_arr = np.asarray(bg_values, dtype=np.float64)
    qc = {
        "step": "background_subtraction",
        "projection_method": method_norm,
        "background_estimator": str(estimator).lower(),
        "background_percentile": _scamp_float(percentile) if str(estimator).lower() == "percentile" else None,
        "roi_area_px": int(np.count_nonzero(mask)),
        "image_height_px": int(data.shape[1]),
        "image_width_px": int(data.shape[2]),
        "z_slices": int(data.shape[0]),
        "z_step_um": _scamp_float(z_step_um),
        "total_z_depth_um": _scamp_float(float(data.shape[0]) * float(z_step_um)) if z_step_um else None,
        "background_slice_mean": _scamp_float(np.mean(bg_arr)) if bg_arr.size else None,
        "background_slice_median": _scamp_float(np.median(bg_arr)) if bg_arr.size else None,
        "background_slice_sd": _scamp_float(np.std(bg_arr)) if bg_arr.size else None,
        "background_slice_min": _scamp_float(np.min(bg_arr)) if bg_arr.size else None,
        "background_slice_max": _scamp_float(np.max(bg_arr)) if bg_arr.size else None,
        "raw_stack_total_intensity": _scamp_float(raw_total),
        "corrected_stack_total_intensity": _scamp_float(corrected_total),
        "removed_intensity_percent": _scamp_float(_scamp_percent_change(raw_total, corrected_total)),
        "projection_total_intensity": _scamp_float(np.sum(proj, dtype=np.float64)),
        "projection_mean_intensity": _scamp_float(np.mean(proj, dtype=np.float64)),
        "projection_max_intensity": _scamp_float(np.max(proj)),
        "clipped_pixel_fraction_percent": _scamp_float(clipped_pixels / float(total_pixels) * 100.0) if total_pixels else None,
    }
    return proj, qc


def _scamp_ask_background_options(self):
    """Modal dialog for background estimator selection. Defaults to median."""
    dlg = tk.Toplevel(self)
    dlg.title("Background subtraction options")
    dlg.transient(self)
    dlg.grab_set()
    dlg.resizable(False, False)

    result = {"ok": False, "estimator": "median", "percentile": 20.0}

    outer = ttk.Frame(dlg, padding=12)
    outer.pack(fill=tk.BOTH, expand=True)
    ttk.Label(outer, text="Background ROI estimator:").pack(anchor=tk.W, pady=(0, 6))

    estimator_var = tk.StringVar(value="median")
    options = [
        ("Median (recommended, robust to bright outliers)", "median"),
        ("Mean (legacy behavior)", "mean"),
        ("Percentile (robust low-background estimate)", "percentile"),
    ]
    for text, value in options:
        ttk.Radiobutton(outer, text=text, value=value, variable=estimator_var).pack(anchor=tk.W, pady=1)

    p_frame = ttk.Frame(outer)
    p_frame.pack(fill=tk.X, pady=(10, 4))
    ttk.Label(p_frame, text="Percentile value:").pack(side=tk.LEFT)
    percentile_var = tk.StringVar(value="20")
    p_entry = ttk.Entry(p_frame, textvariable=percentile_var, width=8)
    p_entry.pack(side=tk.LEFT, padx=(6, 0))
    ttk.Label(p_frame, text="% (used only for Percentile)").pack(side=tk.LEFT, padx=(4, 0))

    note = (
        "The same estimator will be applied to every selected CZI in this batch.\n"
        "QC reports will be saved to qc_reports/ and summarized in the Open files panel."
    )
    ttk.Label(outer, text=note, foreground="#888").pack(anchor=tk.W, pady=(8, 8))

    buttons = ttk.Frame(outer)
    buttons.pack(fill=tk.X)
    def cancel():
        dlg.destroy()
    def ok():
        try:
            p = float(percentile_var.get())
            if p < 0 or p > 100:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid percentile", "Enter a percentile between 0 and 100.", parent=dlg)
            return
        result["ok"] = True
        result["estimator"] = estimator_var.get()
        result["percentile"] = p
        dlg.destroy()
    ttk.Button(buttons, text="Cancel", command=cancel).pack(side=tk.LEFT)
    ttk.Button(buttons, text="Run", command=ok).pack(side=tk.RIGHT)

    dlg.bind("<Return>", lambda _e: ok())
    self.wait_window(dlg)
    return result if result["ok"] else None


def _scamp_qc_short_text(qc):
    """Human-readable one-line QC summary for sidebar/status display."""
    if not qc:
        return ""
    step = qc.get("step", "QC")
    if step == "background_subtraction":
        est = qc.get("background_estimator", "?")
        bg = qc.get("background_slice_mean")
        removed = qc.get("removed_intensity_percent")
        parts = [f"QC bg: {est}"]
        if bg is not None:
            parts.append(f"mean={bg:.2f}")
        if removed is not None:
            parts.append(f"Δ={removed:.1f}%")
        return " | ".join(parts)
    if step == "straightening":
        diff = qc.get("total_intensity_change_percent")
        zero = qc.get("zero_border_fraction_percent")
        parts = ["QC straighten"]
        if diff is not None:
            parts.append(f"Δ={diff:.2f}%")
        if zero is not None:
            parts.append(f"border0={zero:.1f}%")
        return " | ".join(parts)
    return "QC available"


def _scamp_set_editor_qc(ed, qc):
    """Attach QC to an editor and update its visible status when possible."""
    ed.qc_summary = qc
    text = _scamp_qc_short_text(qc)
    try:
        if hasattr(ed, "status_var") and text:
            current = ed.status_var.get()
            if "QC" not in current:
                ed.status_var.set(current + " | " + text)
            else:
                ed.status_var.set(text)
    except Exception:
        pass


def _scamp_zero_border_fraction(arr):
    a = np.asarray(arr)
    if a.ndim != 2 or a.size == 0:
        return None
    borders = [a[0, :], a[-1, :], a[:, 0], a[:, -1]]
    vals = np.concatenate([b.ravel() for b in borders])
    if vals.size == 0:
        return None
    return float(np.count_nonzero(vals <= 0) / vals.size * 100.0)


def _scamp_straighten_qc(source_img, straightened_img, source_name, output_path):
    before = np.asarray(source_img, dtype=np.float64)
    after = np.asarray(straightened_img, dtype=np.float64)
    before_total = float(np.sum(before, dtype=np.float64))
    after_total = float(np.sum(after, dtype=np.float64))
    return {
        "step": "straightening",
        "sample": source_name,
        "output_path": output_path,
        "input_height_px": int(before.shape[0]) if before.ndim == 2 else None,
        "input_width_px": int(before.shape[1]) if before.ndim == 2 else None,
        "output_height_px": int(after.shape[0]) if after.ndim == 2 else None,
        "output_width_px": int(after.shape[1]) if after.ndim == 2 else None,
        "input_total_intensity": _scamp_float(before_total),
        "straightened_total_intensity": _scamp_float(after_total),
        "total_intensity_change_percent": _scamp_float(_scamp_percent_change(before_total, after_total)),
        "input_mean_intensity": _scamp_float(np.mean(before)),
        "straightened_mean_intensity": _scamp_float(np.mean(after)),
        "input_max_intensity": _scamp_float(np.max(before)),
        "straightened_max_intensity": _scamp_float(np.max(after)),
        "output_zero_fraction_percent": _scamp_float(np.count_nonzero(after <= 0) / float(after.size) * 100.0) if after.size else None,
        "zero_border_fraction_percent": _scamp_float(_scamp_zero_border_fraction(after)),
    }


def _scamp_refresh_file_bar_with_qc(self):
    """Sidebar with explicit Load preview button, ROI line, and QC summaries."""
    for child in self.file_list.winfo_children():
        child.destroy()
    for ed in self.editors:
        row = ttk.Frame(self.file_list)
        row.pack(side=tk.TOP, fill=tk.X, padx=4, pady=1)

        top_line = ttk.Frame(row)
        top_line.pack(side=tk.TOP, fill=tk.X)
        label = ed.name[:24] + ("..." if len(ed.name) > 24 else "")
        if getattr(ed, "generated_preview", False):
            label = "↳ " + label
        elif getattr(ed, "is_deferred_czi", False):
            label = "CZI ⏸ " + label
        elif getattr(ed, "is_stack_editor", False):
            label = "Stack ▣ " + label
        ttk.Button(top_line, text=label,
                   command=lambda e=ed: self._select_editor(e)).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(top_line, text="x", width=2, fg="#c0392b",
                  activeforeground="#e74c3c", font=("TkDefaultFont", 10, "bold"),
                  relief=tk.FLAT, bd=0, padx=4, cursor="hand2",
                  command=lambda e=ed: self.close_editor(e)).pack(side=tk.RIGHT, padx=(2, 0))

        if getattr(ed, "is_deferred_czi", False):
            action_line = ttk.Frame(row)
            action_line.pack(side=tk.TOP, fill=tk.X, pady=(1, 1))
            ttk.Button(
                action_line,
                text="Load preview / edit ROI",
                command=lambda e=ed: e.load_full_stack_editor(),
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        roi_path = getattr(ed, "saved_roi_path", None)
        if roi_path and os.path.isfile(roi_path):
            roi_line = ttk.Frame(row)
            roi_line.pack(side=tk.TOP, fill=tk.X, pady=(1, 1))
            ttk.Label(roi_line, text="ROI: " + os.path.basename(roi_path),
                      foreground="#888").pack(side=tk.LEFT, padx=(8, 0))

        qc = getattr(ed, "qc_summary", None)
        if qc:
            qc_line = ttk.Frame(row)
            qc_line.pack(side=tk.TOP, fill=tk.X, pady=(1, 1))
            ttk.Label(qc_line, text=_scamp_qc_short_text(qc),
                      foreground="#7ddc8a").pack(side=tk.LEFT, padx=(8, 0))

        if not getattr(ed, "generated_preview", False):
            cond_line = ttk.Frame(row)
            cond_line.pack(side=tk.TOP, fill=tk.X, pady=(1, 3))
            ttk.Label(cond_line, text="cond:").pack(side=tk.LEFT)
            choices = list(getattr(self, "conditions", []) or [])
            if getattr(ed, "group", "") and ed.group not in choices:
                choices.append(ed.group)
            var = tk.StringVar(value=getattr(ed, "group", "") or (choices[0] if choices else ""))
            combo = ttk.Combobox(cond_line, textvariable=var, values=choices,
                                 state="readonly", width=14)
            combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
            def _on_change(*_a, e=ed, v=var):
                e.group = v.get().strip()
            var.trace_add("write", _on_change)

    try:
        self.update_idletasks()
        self.file_list_canvas.configure(scrollregion=self.file_list_canvas.bbox("all"))
        needed = (self.file_list.winfo_reqheight() > self.file_list_canvas.winfo_height()
                  or len(getattr(self, "editors", [])) > 6)
        if needed and not self.file_list_scrollbar.winfo_ismapped():
            self.file_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, before=self.file_list_canvas)
        elif not needed and self.file_list_scrollbar.winfo_ismapped():
            self.file_list_scrollbar.pack_forget()
            self.file_list_canvas.yview_moveto(0)
    except Exception:
        pass


def _scamp_subtract_background_qc(self):
    """Batch background subtraction with selectable ROI estimator and QC reports."""
    if not self.experiment_dir:
        messagebox.showwarning("No experiment", "No ExperimentID is set.")
        return
    pairs = self._czi_editors_with_rois()
    if not pairs:
        messagebox.showinfo(
            "No CZI + ROI pairs",
            "No open CZI files have an assigned same-name .roi file.\n"
            "Open CZI files, draw a rectangle, then use Tools → Save ROI first."
        )
        return
    if not _HAVE_ROIFILE or not _HAVE_SKDRAW:
        messagebox.showerror(
            "ROI support missing",
            "Background subtraction needs roifile and scikit-image. Update the environment and try again."
        )
        return

    opts = _scamp_ask_background_options(self)
    if opts is None:
        self._log("Background subtraction cancelled.")
        return
    estimator = opts["estimator"]
    percentile = opts["percentile"]

    out_dir = os.path.join(self.experiment_dir, "background_subtracted")
    try:
        os.makedirs(out_dir, exist_ok=True)
        _scamp_qc_dir(self)
    except Exception as exc:
        messagebox.showerror("Output folder error", str(exc))
        return

    fallback_z = None
    created = 0
    skipped = 0
    for ed, roi_path in pairs:
        base = os.path.splitext(os.path.basename(ed.path))[0]
        stack = None
        proj = None
        mask = None
        try:
            self._log(f"Background subtraction: loading {base} ...")
            self.update_idletasks()
            stack, orig_dtype, z_step_meta = load_microscopy_stack(ed.path)
            history = list(getattr(ed, "transform_history", []))
            if history:
                self._log(f"Background subtraction: applying {len(history)} geometry operation(s) for {base} ...")
                self.update_idletasks()
                stack = apply_stack_transform_history(stack, history)
            z_step = getattr(ed, "z_step_um", None) or z_step_meta
            if z_step is None or z_step <= 0:
                if fallback_z is None:
                    fallback_z = self._ask_fallback_z_step()
                    if fallback_z is None:
                        self._log(f"Skipped {base}: no Z-step value available.")
                        skipped += 1
                        continue
                z_step = fallback_z

            roi_rect = getattr(ed, "background_roi_rect", None)
            if roi_rect is not None:
                mask = rectangle_mask_from_rect(roi_rect, stack.shape[1:])
            else:
                mask = load_roi_mask_from_file(roi_path, stack.shape[1:])

            proj, qc = project_stack_with_background_mask_qc(
                stack, mask, method="zdepth", z_step_um=z_step,
                estimator=estimator, percentile=percentile)
            qc.update({
                "sample": base,
                "source_path": ed.path,
                "roi_path": roi_path,
                "condition": getattr(ed, "group", ""),
                "geometry_operations": len(history),
            })

            out_name = build_filename(
                base,
                self.experiment_id,
                ed.group,
                kind="background_subtracted_Zdepth_normalised_sumIP",
                ext=".tif",
            )
            out_path = os.path.join(out_dir, out_name)
            had_signal, _, _ = save_image16(out_path, proj)
            qc["output_path"] = out_path
            qc["had_signal"] = bool(had_signal)
            json_path, csv_path = _scamp_write_qc_report(self, base, "background", qc)
            qc["qc_json"] = json_path
            qc["qc_csv"] = csv_path

            if not had_signal:
                self._log(f"⚠ {ed.name}: background-subtracted projection has no signal. Check the ROI.")

            proj_ed = SampleEditor(
                self.notebook,
                out_path,
                self,
                image_array=proj,
                name=out_name,
                group=ed.group,
            )
            proj_ed.source_base = base
            proj_ed.normalized = proj
            proj_ed.storage_scale_factor = float(qc.get("storage_scale_factor") or SCAMP_ZDEPTH_UINT16_SCALE)
            proj_ed.storage_unit_note = qc.get("storage_unit_note", "")
            _scamp_set_editor_qc(proj_ed, qc)
            _scamp_set_editor_qc(ed, qc)
            self.editors.append(proj_ed)
            self.notebook.add(proj_ed, text=proj_ed.name[:24])
            self.notebook.select(proj_ed)
            created += 1
            self._log(f"Background-subtracted projection → {out_name}")
            self._log(f"QC background report → {json_path}")
        except Exception as exc:
            skipped += 1
            self._log(f"Skipped {base}: background subtraction failed: {exc}")
        finally:
            try:
                del stack
            except Exception:
                pass
            try:
                del mask
            except Exception:
                pass
            try:
                del proj
            except Exception:
                pass
            gc.collect()
            self.update_idletasks()

    self._refresh_file_bar()
    messagebox.showinfo(
        "Background subtraction done",
        f"Created {created} projection(s).\nSkipped {skipped}.\nOutput folder:\n{out_dir}\n\nQC reports:\n{_scamp_qc_dir(self)}",
    )


def _scamp_straighten_current_qc(self):
    """Straighten current image and write per-file QC report."""
    ed = self._current_editor()
    if ed is None:
        self._log("No image loaded.")
        return
    if ed.generated_preview:
        self._log(f"{ed.name}: already a straightened preview file.")
        return
    if not self.experiment_dir:
        messagebox.showwarning("No experiment", "No ExperimentID is set.")
        return
    ok = ed.compute_straighten()
    if not ok:
        self._log(f"{ed.name}: FAILED to straighten")
        return

    str_dir = os.path.join(self.experiment_dir, "straightened")
    try:
        os.makedirs(str_dir, exist_ok=True)
        _scamp_qc_dir(self)
    except Exception as exc:
        self._log(f"{ed.name}: could not create output folder: {exc}")
        return

    str_name = build_filename(ed.source_base, self.experiment_id, ed.group, kind="straightened")
    out_path = os.path.join(str_dir, str_name)

    if getattr(ed, "straightened_path", None) and ed.straightened_path != out_path:
        _remove_image_and_preview(ed.straightened_path)

    if getattr(ed, "preview_editor", None) is not None and ed.preview_editor in self.editors:
        self.close_editor(ed.preview_editor)
        ed.preview_editor = None

    try:
        had_signal, _, _ = save_image16(out_path, ed.straightened)
        if not had_signal:
            self._log(f"⚠ {ed.name}: straightened image has NO signal (all zero) — check the source image and landmarks.")
        ed.straightened_path = out_path

        qc = _scamp_straighten_qc(ed.img, ed.straightened, ed.name, out_path)
        qc.update({
            "sample": ed.source_base,
            "condition": getattr(ed, "group", ""),
            "had_signal": bool(had_signal),
            "source_path": getattr(ed, "path", ""),
        })
        json_path, csv_path = _scamp_write_qc_report(self, ed.source_base, "straighten", qc)
        qc["qc_json"] = json_path
        qc["qc_csv"] = csv_path
        _scamp_set_editor_qc(ed, qc)

        preview = SampleEditor(self.notebook, out_path, self, generated_preview=True)
        preview.straightened = preview.img.copy()
        _scamp_set_editor_qc(preview, qc)
        ed.preview_editor = preview
        self.editors.append(preview)
        self.notebook.add(preview, text=preview.name[:24])
        self.notebook.select(preview)
        self._refresh_file_bar()
        self._log(f"{ed.name}: straightened → {str_name}")
        self._log(f"QC straighten report → {json_path}")
    except Exception as exc:
        self._log(f"{ed.name}: straightened, but could not save/open/QC: {exc}")



# ----------------------------------------------------------------------
#  QC severity labels for preview/sidebar display
# ----------------------------------------------------------------------
def _scamp_qc_value(qc, key, default=None):
    try:
        v = qc.get(key, default)
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _scamp_qc_assess(qc):
    """Return (level, icon, color, reasons) for compact QC display.

    Levels:
        ok      -> green
        warn    -> yellow
        fail    -> red
    """
    if not qc:
        return "ok", "🟢", "#7ddc8a", []

    step = qc.get("step", "")
    reasons = []
    level = "ok"

    def warn(reason):
        nonlocal level
        reasons.append(reason)
        if level == "ok":
            level = "warn"

    def fail(reason):
        nonlocal level
        reasons.append(reason)
        level = "fail"

    if step == "background_subtraction":
        roi_area = _scamp_qc_value(qc, "roi_area_px")
        bg_mean = _scamp_qc_value(qc, "background_slice_mean")
        bg_sd = _scamp_qc_value(qc, "background_slice_sd")
        clipped = _scamp_qc_value(qc, "clipped_pixel_fraction_percent")
        removed = _scamp_qc_value(qc, "removed_intensity_percent")
        proj_total = _scamp_qc_value(qc, "projection_total_intensity")
        had_signal = qc.get("had_signal", True)
        z_step = _scamp_qc_value(qc, "z_step_um")

        if had_signal is False or (proj_total is not None and proj_total <= 0):
            fail("no signal")
        if z_step is None or z_step <= 0:
            warn("missing Z-step")
        if roi_area is not None:
            if roi_area < 200:
                fail("ROI too small")
            elif roi_area < 500:
                warn("small ROI")
        if bg_mean is not None and bg_sd is not None and abs(bg_mean) > 1e-9:
            cv = bg_sd / abs(bg_mean)
            qc["background_slice_cv"] = _scamp_float(cv)
            if cv > 0.50:
                fail("unstable BG")
            elif cv > 0.30:
                warn("variable BG")
        if clipped is not None:
            if clipped > 80:
                fail("very high zero")
            elif clipped > 60:
                warn("high zero")
        if removed is not None:
            if removed < -95:
                fail("too much removed")
            elif removed < -90:
                warn("very strong removal")
            elif removed < -70:
                warn("strong removal")

    elif step == "straightening":
        change = _scamp_qc_value(qc, "total_intensity_change_percent")
        border0 = _scamp_qc_value(qc, "zero_border_fraction_percent")
        out_zero = _scamp_qc_value(qc, "output_zero_fraction_percent")
        had_signal = qc.get("had_signal", True)
        if had_signal is False:
            fail("no signal")
        if change is not None:
            if abs(change) > 5.0:
                fail("intensity changed")
            elif abs(change) > 2.0:
                warn("intensity drift")
        if border0 is not None:
            if border0 > 35:
                fail("large zero border")
            elif border0 > 15:
                warn("zero border")
        if out_zero is not None and out_zero > 70:
            warn("mostly zero output")

    icon = {"ok": "🟢", "warn": "🟡", "fail": "🔴"}.get(level, "🟢")
    color = {"ok": "#7ddc8a", "warn": "#f1c40f", "fail": "#ff6b6b"}.get(level, "#7ddc8a")
    return level, icon, color, reasons


def _scamp_annotate_qc(qc):
    """Add QC level and warnings into the QC dict for reports and display."""
    if not qc:
        return qc
    level, icon, color, reasons = _scamp_qc_assess(qc)
    qc["qc_level"] = level
    qc["qc_icon"] = icon
    qc["qc_status"] = {"ok": "PASS", "warn": "CHECK", "fail": "FAIL"}.get(level, "PASS")
    qc["qc_warnings"] = "; ".join(reasons) if reasons else ""
    return qc


def _scamp_write_qc_report(app, base_name, step_name, qc):
    """Write one QC report as JSON and one-row CSV. Returns (json, csv)."""
    qc = _scamp_annotate_qc(qc)
    qc_dir = _scamp_qc_dir(app)
    safe_base = _sanitize(base_name)
    safe_step = _sanitize(step_name)
    json_path = os.path.join(qc_dir, f"{safe_base}_{safe_step}_qc.json")
    csv_path = os.path.join(qc_dir, f"{safe_base}_{safe_step}_qc.csv")
    payload = dict(qc)
    payload["qc_json"] = json_path
    payload["qc_csv"] = csv_path
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)
    return json_path, csv_path


def _scamp_qc_short_text(qc):
    """Compact QC text for preview/sidebar display, with severity icon."""
    if not qc:
        return ""
    qc = _scamp_annotate_qc(qc)
    icon = qc.get("qc_icon", "🟢")
    step = qc.get("step", "QC")

    if step == "background_subtraction":
        est = str(qc.get("background_estimator", "?")).capitalize()
        bg = qc.get("background_slice_mean")
        removed = qc.get("removed_intensity_percent")
        zero = qc.get("clipped_pixel_fraction_percent")
        roi = qc.get("roi_area_px")
        status = {"ok": "PASS", "warn": "CHECK", "fail": "FAIL"}.get(qc.get("qc_level", "ok"), "PASS")
        parts = [f"QC BG {icon} {status}", f"Method: {est}"]
        if roi is not None:
            parts.append(f"ROI: {int(roi)} px")
        if bg is not None:
            parts.append(f"BG: {float(bg):.2f}")
        if removed is not None:
            parts.append(f"Δ: {float(removed):.0f}%")
        if zero is not None:
            parts.append(f"Zero: {float(zero):.0f}%")
        warnings = qc.get("qc_warnings")
        if warnings:
            parts.append(f"⚠ {warnings}")
        return " | ".join(parts)

    if step == "straightening":
        change = qc.get("total_intensity_change_percent")
        border0 = qc.get("zero_border_fraction_percent")
        out_zero = qc.get("output_zero_fraction_percent")
        status = {"ok": "PASS", "warn": "CHECK", "fail": "FAIL"}.get(qc.get("qc_level", "ok"), "PASS")
        parts = [f"QC Straighten {icon} {status}"]
        if change is not None:
            parts.append(f"Δ: {float(change):+.2f}%")
        if border0 is not None:
            parts.append(f"Border0: {float(border0):.0f}%")
        if out_zero is not None:
            parts.append(f"Zero: {float(out_zero):.0f}%")
        warnings = qc.get("qc_warnings")
        if warnings:
            parts.append(f"⚠ {warnings}")
        return " | ".join(parts)

    return f"QC {icon} available"


def _scamp_set_editor_qc(ed, qc):
    """Attach QC to an editor and update its visible status when possible."""
    qc = _scamp_annotate_qc(qc)
    ed.qc_summary = qc
    text = _scamp_qc_short_text(qc)
    try:
        if hasattr(ed, "status_var") and text:
            current = ed.status_var.get()
            base = current.split(" | QC ")[0] if " | QC " in current else current
            ed.status_var.set(base + " | " + text)
    except Exception:
        pass


def _scamp_refresh_file_bar_with_qc(self):
    """Sidebar with explicit Load preview button, ROI line, and colored QC summaries."""
    for child in self.file_list.winfo_children():
        child.destroy()
    for ed in self.editors:
        row = ttk.Frame(self.file_list)
        row.pack(side=tk.TOP, fill=tk.X, padx=4, pady=1)

        top_line = ttk.Frame(row)
        top_line.pack(side=tk.TOP, fill=tk.X)
        label = ed.name[:24] + ("..." if len(ed.name) > 24 else "")
        if getattr(ed, "generated_preview", False):
            label = "↳ " + label
        elif getattr(ed, "is_deferred_czi", False):
            label = "CZI ⏸ " + label
        elif getattr(ed, "is_stack_editor", False):
            label = "Stack ▣ " + label
        ttk.Button(top_line, text=label,
                   command=lambda e=ed: self._select_editor(e)).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(top_line, text="x", width=2, fg="#c0392b",
                  activeforeground="#e74c3c", font=("TkDefaultFont", 10, "bold"),
                  relief=tk.FLAT, bd=0, padx=4, cursor="hand2",
                  command=lambda e=ed: self.close_editor(e)).pack(side=tk.RIGHT, padx=(2, 0))

        if getattr(ed, "is_deferred_czi", False):
            action_line = ttk.Frame(row)
            action_line.pack(side=tk.TOP, fill=tk.X, pady=(1, 1))
            ttk.Button(
                action_line,
                text="Load preview / edit ROI",
                command=lambda e=ed: e.load_full_stack_editor(),
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        roi_path = getattr(ed, "saved_roi_path", None)
        if roi_path and os.path.isfile(roi_path):
            roi_line = ttk.Frame(row)
            roi_line.pack(side=tk.TOP, fill=tk.X, pady=(1, 1))
            ttk.Label(roi_line, text="ROI: " + os.path.basename(roi_path),
                      foreground="#888").pack(side=tk.LEFT, padx=(8, 0))

        qc = getattr(ed, "qc_summary", None)
        if qc:
            qc = _scamp_annotate_qc(qc)
            _level, _icon, color, _reasons = _scamp_qc_assess(qc)
            qc_line = ttk.Frame(row)
            qc_line.pack(side=tk.TOP, fill=tk.X, pady=(1, 1))
            ttk.Label(qc_line, text=_scamp_qc_short_text(qc),
                      foreground=color).pack(side=tk.LEFT, padx=(8, 0))

        if not getattr(ed, "generated_preview", False):
            cond_line = ttk.Frame(row)
            cond_line.pack(side=tk.TOP, fill=tk.X, pady=(1, 3))
            ttk.Label(cond_line, text="cond:").pack(side=tk.LEFT)
            choices = list(getattr(self, "conditions", []) or [])
            if getattr(ed, "group", "") and ed.group not in choices:
                choices.append(ed.group)
            var = tk.StringVar(value=getattr(ed, "group", "") or (choices[0] if choices else ""))
            combo = ttk.Combobox(cond_line, textvariable=var, values=choices,
                                 state="readonly", width=14)
            combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
            def _on_change(*_a, e=ed, v=var):
                e.group = v.get().strip()
            var.trace_add("write", _on_change)

    try:
        self.update_idletasks()
        self.file_list_canvas.configure(scrollregion=self.file_list_canvas.bbox("all"))
        needed = (self.file_list.winfo_reqheight() > self.file_list_canvas.winfo_height()
                  or len(getattr(self, "editors", [])) > 6)
        if needed and not self.file_list_scrollbar.winfo_ismapped():
            self.file_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, before=self.file_list_canvas)
        elif not needed and self.file_list_scrollbar.winfo_ismapped():
            self.file_list_scrollbar.pack_forget()
            self.file_list_canvas.yview_moveto(0)
    except Exception:
        pass



# ----------------------------------------------------------------------
#  SCAMP QC v2: biology-aware background QC + clickable QC details
# ----------------------------------------------------------------------
def project_stack_with_background_mask_qc(stack, mask, method="zdepth", z_step_um=None,
                                          estimator="median", percentile=20.0):
    """Memory-light background-corrected projection plus biology-aware QC.

    Zero/clipped pixels are still reported, but they no longer drive QC severity
    by themselves because sparse calcification images are expected to contain a
    large background area after subtraction.
    """
    data = np.asarray(stack, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("Expected a ZYX stack.")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != data.shape[1:]:
        raise ValueError(
            f"ROI mask shape {mask.shape} does not match image shape {data.shape[1:]}.")
    if not mask.any():
        raise ValueError("ROI mask is empty.")

    method_norm = str(method).lower()
    if method_norm not in ("mean", "sum", "zdepth", "z-depth", "zdepth normalised sum"):
        raise ValueError(f"Unknown projection method: {method}")

    # Signal-reference values from the raw stack. These are intentionally based
    # on high percentiles rather than max intensity, because max is dominated by
    # hot pixels and rare artifacts.
    finite_raw = data[np.isfinite(data)]
    raw_signal95 = _scamp_float(np.percentile(finite_raw, 95)) if finite_raw.size else None
    raw_signal99 = _scamp_float(np.percentile(finite_raw, 99)) if finite_raw.size else None

    acc = np.zeros(data.shape[1:], dtype=np.float32)
    bg_values = []
    raw_total = 0.0
    corrected_total = 0.0
    clipped_pixels = 0
    total_pixels = int(data.size)
    positive_after_pixels = 0

    for z in range(data.shape[0]):
        sl = data[z].astype(np.float32, copy=False)
        bg = _scamp_estimate_background(sl[mask], estimator=estimator, percentile=percentile)
        bg_values.append(bg)
        raw_total += float(np.sum(sl, dtype=np.float64))
        corr = sl - bg
        clipped_pixels += int(np.count_nonzero(corr < 0))
        np.maximum(corr, 0, out=corr)
        positive_after_pixels += int(np.count_nonzero(corr > 0))
        corrected_total += float(np.sum(corr, dtype=np.float64))
        acc += corr

    if method_norm == "mean":
        proj = acc / float(data.shape[0])
    elif method_norm in ("zdepth", "z-depth", "zdepth normalised sum"):
        if z_step_um is None or z_step_um <= 0:
            raise ValueError("Z-depth normalised sum needs a positive Z-step in µm.")
        depth_um = float(data.shape[0]) * float(z_step_um)
        proj = acc / depth_um
    else:
        proj = acc

    bg_arr = np.asarray(bg_values, dtype=np.float64)
    bg_mean = float(np.mean(bg_arr)) if bg_arr.size else None
    bg_to_signal95 = None
    if bg_mean is not None and raw_signal95 is not None and abs(float(raw_signal95)) > 1e-9:
        bg_to_signal95 = float(bg_mean) / float(raw_signal95) * 100.0
    bg_to_signal99 = None
    if bg_mean is not None and raw_signal99 is not None and abs(float(raw_signal99)) > 1e-9:
        bg_to_signal99 = float(bg_mean) / float(raw_signal99) * 100.0

    retained_intensity_percent = None
    if abs(raw_total) > 1e-12:
        retained_intensity_percent = corrected_total / raw_total * 100.0

    positive_after_fraction = positive_after_pixels / float(total_pixels) * 100.0 if total_pixels else None
    proj_finite = proj[np.isfinite(proj)]
    projection_signal95 = _scamp_float(np.percentile(proj_finite, 95)) if proj_finite.size else None
    projection_signal99 = _scamp_float(np.percentile(proj_finite, 99)) if proj_finite.size else None

    qc = {
        "step": "background_subtraction",
        "projection_method": method_norm,
        "background_estimator": str(estimator).lower(),
        "background_percentile": _scamp_float(percentile) if str(estimator).lower() == "percentile" else None,
        "roi_area_px": int(np.count_nonzero(mask)),
        "image_height_px": int(data.shape[1]),
        "image_width_px": int(data.shape[2]),
        "z_slices": int(data.shape[0]),
        "z_step_um": _scamp_float(z_step_um),
        "total_z_depth_um": _scamp_float(float(data.shape[0]) * float(z_step_um)) if z_step_um else None,
        "background_slice_mean": _scamp_float(bg_mean),
        "background_slice_median": _scamp_float(np.median(bg_arr)) if bg_arr.size else None,
        "background_slice_sd": _scamp_float(np.std(bg_arr)) if bg_arr.size else None,
        "background_slice_min": _scamp_float(np.min(bg_arr)) if bg_arr.size else None,
        "background_slice_max": _scamp_float(np.max(bg_arr)) if bg_arr.size else None,
        "raw_stack_total_intensity": _scamp_float(raw_total),
        "corrected_stack_total_intensity": _scamp_float(corrected_total),
        "removed_intensity_percent": _scamp_float(_scamp_percent_change(raw_total, corrected_total)),
        "retained_intensity_percent": _scamp_float(retained_intensity_percent),
        "raw_stack_signal95": _scamp_float(raw_signal95),
        "raw_stack_signal99": _scamp_float(raw_signal99),
        "background_to_signal95_percent": _scamp_float(bg_to_signal95),
        "background_to_signal99_percent": _scamp_float(bg_to_signal99),
        "projection_total_intensity": _scamp_float(np.sum(proj, dtype=np.float64)),
        "projection_mean_intensity": _scamp_float(np.mean(proj, dtype=np.float64)),
        "projection_max_intensity": _scamp_float(np.max(proj)),
        "projection_signal95": projection_signal95,
        "projection_signal99": projection_signal99,
        "clipped_pixel_fraction_percent": _scamp_float(clipped_pixels / float(total_pixels) * 100.0) if total_pixels else None,
        "positive_after_fraction_percent": _scamp_float(positive_after_fraction),
        "zero_fraction_note": (
            "High zero fraction can be normal for sparse calcification images; "
            "it is reported but does not trigger QC warnings by itself."
        ),
    }
    return proj, qc


def _scamp_qc_assess(qc):
    """Return (level, icon, color, reasons) for compact QC display.

    Conservative QC policy:
    - FAIL means a technical problem that likely makes the output unusable.
    - CHECK means the output exists, but a human should inspect it.
    - PASS means no obvious technical issue was detected.

    Biology-dependent metrics such as strong background removal, high zero
    fraction, or high BG/signal ratios are CHECK only. They do not trigger FAIL
    by themselves, because sparse calcification images can naturally contain a
    large background area and a small high-signal mineralized region.
    """
    if not qc:
        return "ok", "🟢", "#7ddc8a", []

    step = qc.get("step", "")
    reasons = []
    level = "ok"

    def warn(reason):
        nonlocal level
        reasons.append(reason)
        if level == "ok":
            level = "warn"

    def fail(reason):
        nonlocal level
        reasons.append(reason)
        level = "fail"

    def bad_number(value):
        try:
            v = float(value)
            return not np.isfinite(v)
        except Exception:
            return value is not None

    if step == "background_subtraction":
        roi_area = _scamp_qc_value(qc, "roi_area_px")
        bg_mean = _scamp_qc_value(qc, "background_slice_mean")
        bg_sd = _scamp_qc_value(qc, "background_slice_sd")
        bg_sig95 = _scamp_qc_value(qc, "background_to_signal95_percent")
        bg_sig99 = _scamp_qc_value(qc, "background_to_signal99_percent")
        proj_total = _scamp_qc_value(qc, "projection_total_intensity")
        proj_max = _scamp_qc_value(qc, "projection_max_intensity")
        proj_sig99 = _scamp_qc_value(qc, "projection_signal99")
        retained = _scamp_qc_value(qc, "retained_intensity_percent")
        positive_after = _scamp_qc_value(qc, "positive_after_fraction_percent")
        had_signal = qc.get("had_signal", True)
        z_step = _scamp_qc_value(qc, "z_step_um")
        method = str(qc.get("projection_method", "")).lower()

        # Hard technical failures only.
        if had_signal is False:
            fail("no signal")
        if proj_total is not None and proj_total <= 0:
            fail("empty projection")
        if proj_max is not None and proj_max <= 0:
            fail("all-zero projection")
        if method in ("zdepth", "z-depth", "zdepth normalised sum") and (z_step is None or z_step <= 0):
            fail("missing Z-step")
        if roi_area is None or roi_area <= 0:
            fail("no ROI")
        elif roi_area < 100:
            fail("ROI too small")
        elif roi_area < 500:
            warn("small ROI")

        for key in (
            "background_slice_mean", "background_slice_sd",
            "projection_total_intensity", "projection_max_intensity",
            "raw_stack_total_intensity", "corrected_stack_total_intensity",
        ):
            if bad_number(qc.get(key)):
                fail(f"invalid {key}")

        # Human-check warnings. These are not fatal because they depend on image
        # type, signal sparsity, and biological background.
        if bg_mean is not None and bg_sd is not None and abs(bg_mean) > 1e-9:
            cv = bg_sd / abs(bg_mean)
            qc["background_slice_cv"] = _scamp_float(cv)
            if cv > 0.50:
                warn("unstable BG")
            elif cv > 0.30:
                warn("variable BG")

        if bg_sig99 is not None:
            if bg_sig99 > 60:
                warn("BG very high vs signal99")
            elif bg_sig99 > 40:
                warn("BG high vs signal99")
            elif bg_sig99 > 20:
                warn("BG moderate vs signal99")
        elif bg_sig95 is not None:
            # signal95 is too conservative for sparse mineralization, so it is
            # used only as a soft CHECK fallback.
            if bg_sig95 > 75:
                warn("BG high vs signal95")
            elif bg_sig95 > 50:
                warn("BG moderate vs signal95")

        if retained is not None:
            if retained < 5:
                warn("very low retained intensity")
            elif retained < 15:
                warn("low retained intensity")

        if positive_after is not None and positive_after < 0.5:
            warn("very sparse positive signal")

        # Deliberately do not warn/fail from zero/clipped fraction alone.

    elif step == "straightening":
        change = _scamp_qc_value(qc, "total_intensity_change_percent")
        border0 = _scamp_qc_value(qc, "zero_border_fraction_percent")
        out_zero = _scamp_qc_value(qc, "output_zero_fraction_percent")
        had_signal = qc.get("had_signal", True)
        if had_signal is False:
            fail("no signal")
        if change is not None:
            # Straightening should be much more intensity-preserving than
            # background subtraction. Only very large changes are FAIL.
            if abs(change) > 20.0:
                fail("large intensity change")
            elif abs(change) > 5.0:
                warn("intensity drift")
            elif abs(change) > 2.0:
                warn("minor intensity drift")
        if border0 is not None:
            if border0 > 50:
                warn("large zero border")
            elif border0 > 20:
                warn("zero border")
        if out_zero is not None and out_zero > 95:
            fail("empty straightened output")
        elif out_zero is not None and out_zero > 80:
            warn("mostly zero output")

        for key in ("total_intensity_before", "total_intensity_after", "total_intensity_change_percent"):
            if bad_number(qc.get(key)):
                fail(f"invalid {key}")

    icon = {"ok": "🟢", "warn": "🟡", "fail": "🔴"}.get(level, "🟢")
    color = {"ok": "#7ddc8a", "warn": "#f1c40f", "fail": "#ff6b6b"}.get(level, "#7ddc8a")
    return level, icon, color, reasons

def _scamp_qc_short_text(qc):
    """Compact QC text for preview/sidebar display, with severity icon."""
    if not qc:
        return ""
    qc = _scamp_annotate_qc(qc)
    icon = qc.get("qc_icon", "🟢")
    step = qc.get("step", "QC")

    if step == "background_subtraction":
        est = str(qc.get("background_estimator", "?")).capitalize()
        bg = qc.get("background_slice_mean")
        bg_sig95 = qc.get("background_to_signal95_percent")
        bg_sig99 = qc.get("background_to_signal99_percent")
        retained = qc.get("retained_intensity_percent")
        pos_after = qc.get("positive_after_fraction_percent")
        roi = qc.get("roi_area_px")
        status = {"ok": "PASS", "warn": "CHECK", "fail": "FAIL"}.get(qc.get("qc_level", "ok"), "PASS")
        parts = [f"QC BG {icon} {status}", f"Method: {est}"]
        if roi is not None:
            parts.append(f"ROI: {int(roi)} px")
        if bg is not None:
            parts.append(f"BG: {float(bg):.2f}")
        if bg_sig99 is not None:
            parts.append(f"BG/S99: {float(bg_sig99):.1f}%")
        elif bg_sig95 is not None:
            parts.append(f"BG/S95: {float(bg_sig95):.1f}%")
        if retained is not None:
            parts.append(f"Retained: {float(retained):.0f}%")
        if pos_after is not None:
            parts.append(f"Pos: {float(pos_after):.0f}%")
        warnings = qc.get("qc_warnings")
        if warnings:
            parts.append(f"⚠ {warnings}")
        return " | ".join(parts)

    if step == "straightening":
        change = qc.get("total_intensity_change_percent")
        border0 = qc.get("zero_border_fraction_percent")
        status = {"ok": "PASS", "warn": "CHECK", "fail": "FAIL"}.get(qc.get("qc_level", "ok"), "PASS")
        parts = [f"QC Straighten {icon} {status}"]
        if change is not None:
            parts.append(f"Δ: {float(change):+.2f}%")
        if border0 is not None:
            parts.append(f"Border0: {float(border0):.0f}%")
        warnings = qc.get("qc_warnings")
        if warnings:
            parts.append(f"⚠ {warnings}")
        return " | ".join(parts)

    return f"QC {icon} available"


def _scamp_show_qc_details(app_or_widget, qc, title="QC details"):
    """Open a simple read-only QC details window."""
    if not qc:
        return
    qc = _scamp_annotate_qc(dict(qc))
    parent = app_or_widget
    try:
        if not isinstance(parent, tk.Tk) and not isinstance(parent, tk.Toplevel):
            parent = parent.winfo_toplevel()
    except Exception:
        parent = None

    dlg = tk.Toplevel(parent) if parent is not None else tk.Toplevel()
    dlg.title(title)
    dlg.geometry("760x520")
    try:
        dlg.transient(parent)
    except Exception:
        pass

    outer = ttk.Frame(dlg, padding=10)
    outer.pack(fill=tk.BOTH, expand=True)

    header = _scamp_qc_short_text(qc)
    _level, _icon, color, _reasons = _scamp_qc_assess(qc)
    ttk.Label(outer, text=header, foreground=color, font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, pady=(0, 8))

    text = tk.Text(outer, wrap="word", height=22)
    scroll = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=text.yview)
    text.configure(yscrollcommand=scroll.set)
    text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)

    preferred = [
        "step", "sample", "condition", "qc_level", "qc_warnings",
        "background_estimator", "background_percentile", "roi_area_px",
        "background_slice_mean", "background_slice_median", "background_slice_sd", "background_slice_cv",
        "raw_stack_signal95", "raw_stack_signal99", "background_to_signal95_percent", "background_to_signal99_percent",
        "raw_stack_total_intensity", "corrected_stack_total_intensity", "retained_intensity_percent", "removed_intensity_percent",
        "clipped_pixel_fraction_percent", "positive_after_fraction_percent", "zero_fraction_note",
        "projection_total_intensity", "projection_mean_intensity", "projection_max_intensity", "projection_signal95", "projection_signal99",
        "z_slices", "z_step_um", "total_z_depth_um",
        "input_total_intensity", "straightened_total_intensity", "total_intensity_change_percent",
        "input_mean_intensity", "straightened_mean_intensity", "input_max_intensity", "straightened_max_intensity",
        "zero_border_fraction_percent", "output_zero_fraction_percent",
        "source_path", "output_path", "qc_json", "qc_csv",
    ]
    lines = []
    seen = set()
    for key in preferred:
        if key in qc:
            lines.append(f"{key}: {qc.get(key)}")
            seen.add(key)
    for key in sorted(k for k in qc.keys() if k not in seen):
        lines.append(f"{key}: {qc.get(key)}")
    text.insert("1.0", "\n".join(lines))
    text.configure(state="disabled")

    bottom = ttk.Frame(dlg, padding=(10, 0, 10, 10))
    bottom.pack(side=tk.BOTTOM, fill=tk.X)
    ttk.Button(bottom, text="Close", command=dlg.destroy).pack(side=tk.RIGHT)


def _scamp_attach_editor_qc_click(ed, qc):
    """Add/update a clickable QC details control inside an editor preview."""
    try:
        if getattr(ed, "_scamp_qc_button", None) is not None and ed._scamp_qc_button.winfo_exists():
            ed._scamp_qc_button.configure(text="QC details", command=lambda e=ed, q=qc: _scamp_show_qc_details(e, q))
            return
    except Exception:
        pass
    try:
        # In both SampleEditor and StackEditor the first child is the top toolbar.
        children = ed.winfo_children()
        toolbar = children[0] if children else ed
        btn = ttk.Button(toolbar, text="QC details", command=lambda e=ed, q=qc: _scamp_show_qc_details(e, q))
        btn.pack(side=tk.RIGHT, padx=3, pady=2)
        ed._scamp_qc_button = btn
    except Exception:
        pass


def _scamp_set_editor_qc(ed, qc):
    """Attach QC to an editor and update visible/clickable QC status."""
    qc = _scamp_annotate_qc(qc)
    ed.qc_summary = qc
    text = _scamp_qc_short_text(qc)
    try:
        if hasattr(ed, "status_var") and text:
            current = ed.status_var.get()
            base = current.split(" | QC ")[0] if " | QC " in current else current
            ed.status_var.set(base + " | " + text + "  (click QC details)")
    except Exception:
        pass
    _scamp_attach_editor_qc_click(ed, qc)


def _scamp_refresh_file_bar_with_qc(self):
    """Sidebar with clickable QC summaries."""
    for child in self.file_list.winfo_children():
        child.destroy()
    for ed in self.editors:
        row = ttk.Frame(self.file_list)
        row.pack(side=tk.TOP, fill=tk.X, padx=4, pady=1)

        top_line = ttk.Frame(row)
        top_line.pack(side=tk.TOP, fill=tk.X)
        label = ed.name[:24] + ("..." if len(ed.name) > 24 else "")
        if getattr(ed, "generated_preview", False):
            label = "↳ " + label
        elif getattr(ed, "is_deferred_czi", False):
            label = "CZI ⏸ " + label
        elif getattr(ed, "is_stack_editor", False):
            label = "Stack ▣ " + label
        ttk.Button(top_line, text=label,
                   command=lambda e=ed: self._select_editor(e)).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(top_line, text="x", width=2, fg="#c0392b",
                  activeforeground="#e74c3c", font=("TkDefaultFont", 10, "bold"),
                  relief=tk.FLAT, bd=0, padx=4, cursor="hand2",
                  command=lambda e=ed: self.close_editor(e)).pack(side=tk.RIGHT, padx=(2, 0))

        if getattr(ed, "is_deferred_czi", False):
            action_line = ttk.Frame(row)
            action_line.pack(side=tk.TOP, fill=tk.X, pady=(1, 1))
            ttk.Button(
                action_line,
                text="Load preview / edit ROI",
                command=lambda e=ed: e.load_full_stack_editor(),
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        roi_path = getattr(ed, "saved_roi_path", None)
        if roi_path and os.path.isfile(roi_path):
            roi_line = ttk.Frame(row)
            roi_line.pack(side=tk.TOP, fill=tk.X, pady=(1, 1))
            ttk.Label(roi_line, text="ROI: " + os.path.basename(roi_path),
                      foreground="#888").pack(side=tk.LEFT, padx=(8, 0))

        qc = getattr(ed, "qc_summary", None)
        if qc:
            qc = _scamp_annotate_qc(qc)
            _level, _icon, color, _reasons = _scamp_qc_assess(qc)
            qc_line = ttk.Frame(row)
            qc_line.pack(side=tk.TOP, fill=tk.X, pady=(1, 1))
            qc_text = _scamp_qc_short_text(qc) + "  [details]"
            qc_label = ttk.Label(qc_line, text=qc_text, foreground=color, cursor="hand2")
            qc_label.pack(side=tk.LEFT, padx=(8, 0))
            qc_label.bind("<Button-1>", lambda _ev, q=qc, e=ed: _scamp_show_qc_details(e, q))

        if not getattr(ed, "generated_preview", False):
            cond_line = ttk.Frame(row)
            cond_line.pack(side=tk.TOP, fill=tk.X, pady=(1, 3))
            ttk.Label(cond_line, text="cond:").pack(side=tk.LEFT)
            choices = list(getattr(self, "conditions", []) or [])
            if getattr(ed, "group", "") and ed.group not in choices:
                choices.append(ed.group)
            var = tk.StringVar(value=getattr(ed, "group", "") or (choices[0] if choices else ""))
            combo = ttk.Combobox(cond_line, textvariable=var, values=choices,
                                 state="readonly", width=14)
            combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
            def _on_change(*_a, e=ed, v=var):
                e.group = v.get().strip()
            var.trace_add("write", _on_change)

    try:
        self.update_idletasks()
        self.file_list_canvas.configure(scrollregion=self.file_list_canvas.bbox("all"))
        needed = (self.file_list.winfo_reqheight() > self.file_list_canvas.winfo_height()
                  or len(getattr(self, "editors", [])) > 6)
        if needed and not self.file_list_scrollbar.winfo_ismapped():
            self.file_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, before=self.file_list_canvas)
        elif not needed and self.file_list_scrollbar.winfo_ismapped():
            self.file_list_scrollbar.pack_forget()
            self.file_list_canvas.yview_moveto(0)
    except Exception:
        pass

# Install QC-aware handlers last.
App._refresh_file_bar = _scamp_refresh_file_bar_with_qc
App.subtract_background_for_czi_with_rois = _scamp_subtract_background_qc
App.straighten_current = _scamp_straighten_current_qc


# ======================================================================
#  SCAMP QC v3: corridor-aware straightening QC
# ======================================================================
def _scamp_source_corridor_mask_from_guides(editor):
    """Return a boolean mask for the source spine corridor defined by guides.

    The previous straightening QC compared the whole input image to the
    straightened output. That is not meaningful because straightening extracts
    only the guide-defined spine corridor. This mask approximates that source
    corridor by filling the polygon between the interpolated dorsal and ventral
    guide boundaries.
    """
    if editor is None or not getattr(editor, "has_straighten_guides", False):
        return None
    triplets = getattr(editor, "triplets", None)
    if not triplets or len(triplets) < 2:
        return None
    above, below = interpolate_boundaries(triplets, num_interpolated=7)
    if above is None or below is None or len(above) < 3 or len(below) < 3:
        return None
    try:
        pts = np.vstack([above, below[::-1]])
        h, w = np.asarray(editor.img).shape[:2]
        rr, cc = _sk_polygon(pts[:, 1], pts[:, 0], shape=(h, w))
        mask = np.zeros((h, w), dtype=bool)
        mask[rr, cc] = True
        if not mask.any():
            return None
        return mask
    except Exception:
        return None


def _scamp_straighten_qc_corridor(editor, straightened_img, source_name, output_path):
    """Create straightening QC using the guide-defined source corridor.

    The primary intensity-conservation metric is now:
        source corridor total intensity vs straightened total intensity
    not whole input image vs straightened total intensity.
    """
    before = np.asarray(getattr(editor, "img", None), dtype=np.float64)
    after = np.asarray(straightened_img, dtype=np.float64)
    whole_total = float(np.sum(before, dtype=np.float64)) if before.size else 0.0
    after_total = float(np.sum(after, dtype=np.float64)) if after.size else 0.0

    mask = _scamp_source_corridor_mask_from_guides(editor)
    corridor_total = None
    corridor_area = None
    corridor_mean = None
    corridor_max = None
    corridor_change = None
    corridor_to_whole = None
    if mask is not None and before.ndim == 2 and mask.shape == before.shape:
        vals = before[mask]
        corridor_area = int(mask.sum())
        corridor_total = float(np.sum(vals, dtype=np.float64))
        corridor_mean = float(np.mean(vals)) if vals.size else None
        corridor_max = float(np.max(vals)) if vals.size else None
        corridor_change = _scamp_percent_change(corridor_total, after_total)
        corridor_to_whole = (corridor_total / whole_total * 100.0) if whole_total > 0 else None

    # Keep the old whole-image value for context only. It must not drive FAIL.
    whole_change = _scamp_percent_change(whole_total, after_total)
    out_zero = (np.count_nonzero(after <= 0) / float(after.size) * 100.0) if after.size else None
    border_zero = _scamp_zero_border_fraction(after)
    max_change = None
    if corridor_max is not None and corridor_max > 0 and after.size:
        max_change = _scamp_percent_change(corridor_max, float(np.max(after)))

    return {
        "step": "straightening",
        "sample": source_name,
        "output_path": output_path,
        "input_height_px": int(before.shape[0]) if before.ndim == 2 else None,
        "input_width_px": int(before.shape[1]) if before.ndim == 2 else None,
        "output_height_px": int(after.shape[0]) if after.ndim == 2 else None,
        "output_width_px": int(after.shape[1]) if after.ndim == 2 else None,
        "input_total_intensity_whole_image": _scamp_float(whole_total),
        "whole_image_to_straightened_change_percent_context_only": _scamp_float(whole_change),
        "source_corridor_mask_found": bool(mask is not None),
        "source_corridor_area_px": corridor_area,
        "source_corridor_fraction_of_image_percent": _scamp_float((corridor_area / before.size * 100.0) if corridor_area and before.size else None),
        "source_corridor_fraction_of_whole_intensity_percent": _scamp_float(corridor_to_whole),
        "source_corridor_total_intensity": _scamp_float(corridor_total),
        "straightened_total_intensity": _scamp_float(after_total),
        "corridor_intensity_change_percent": _scamp_float(corridor_change),
        "source_corridor_mean_intensity": _scamp_float(corridor_mean),
        "straightened_mean_intensity": _scamp_float(np.mean(after)) if after.size else None,
        "source_corridor_max_intensity": _scamp_float(corridor_max),
        "straightened_max_intensity": _scamp_float(np.max(after)) if after.size else None,
        "max_intensity_change_percent_context_only": _scamp_float(max_change),
        "output_zero_fraction_percent": _scamp_float(out_zero),
        "zero_border_fraction_percent": _scamp_float(border_zero),
        "guide_count": int(len(getattr(editor, "triplets", []) or [])),
        "qc_note": (
            "Straightening QC compares the guide-defined source spine corridor "
            "against the straightened output. Whole-image intensity change is "
            "reported only as context because the input image contains large "
            "areas outside the straightened corridor."
        ),
    }


def _scamp_qc_assess(qc):
    """Return (level, icon, color, reasons) using conservative QC rules.

    FAIL is reserved for technical problems that likely make the output
    unusable. CHECK means the output exists but should be visually inspected.
    """
    if not qc:
        return "ok", "🟢", "#7ddc8a", []

    step = qc.get("step", "")
    reasons = []
    level = "ok"

    def warn(reason):
        nonlocal level
        reasons.append(reason)
        if level == "ok":
            level = "warn"

    def fail(reason):
        nonlocal level
        reasons.append(reason)
        level = "fail"

    def bad_number(value):
        try:
            v = float(value)
            return not np.isfinite(v)
        except Exception:
            return value is not None

    if step == "background_subtraction":
        roi_area = _scamp_qc_value(qc, "roi_area_px")
        bg_mean = _scamp_qc_value(qc, "background_slice_mean")
        bg_sd = _scamp_qc_value(qc, "background_slice_sd")
        bg_sig95 = _scamp_qc_value(qc, "background_to_signal95_percent")
        bg_sig99 = _scamp_qc_value(qc, "background_to_signal99_percent")
        proj_total = _scamp_qc_value(qc, "projection_total_intensity")
        proj_max = _scamp_qc_value(qc, "projection_max_intensity")
        retained = _scamp_qc_value(qc, "retained_intensity_percent")
        positive_after = _scamp_qc_value(qc, "positive_after_fraction_percent")
        had_signal = qc.get("had_signal", True)
        z_step = _scamp_qc_value(qc, "z_step_um")
        method = str(qc.get("projection_method", "")).lower()

        if had_signal is False:
            fail("no signal")
        if proj_total is not None and proj_total <= 0:
            fail("empty projection")
        if proj_max is not None and proj_max <= 0:
            fail("all-zero projection")
        if method in ("zdepth", "z-depth", "zdepth normalised sum") and (z_step is None or z_step <= 0):
            fail("missing Z-step")
        if roi_area is None or roi_area <= 0:
            fail("no ROI")
        elif roi_area < 100:
            fail("ROI too small")
        elif roi_area < 500:
            warn("small ROI")

        for key in (
            "background_slice_mean", "background_slice_sd",
            "projection_total_intensity", "projection_max_intensity",
            "raw_stack_total_intensity", "corrected_stack_total_intensity",
        ):
            if bad_number(qc.get(key)):
                fail(f"invalid {key}")

        if bg_mean is not None and bg_sd is not None and abs(bg_mean) > 1e-9:
            cv = bg_sd / abs(bg_mean)
            qc["background_slice_cv"] = _scamp_float(cv)
            if cv > 0.50:
                warn("unstable BG")
            elif cv > 0.30:
                warn("variable BG")

        if bg_sig99 is not None:
            if bg_sig99 > 60:
                warn("BG very high vs signal99")
            elif bg_sig99 > 40:
                warn("BG high vs signal99")
            elif bg_sig99 > 20:
                warn("BG moderate vs signal99")
        elif bg_sig95 is not None:
            if bg_sig95 > 75:
                warn("BG high vs signal95")
            elif bg_sig95 > 50:
                warn("BG moderate vs signal95")

        if retained is not None:
            if retained < 5:
                warn("very low retained intensity")
            elif retained < 15:
                warn("low retained intensity")

        if positive_after is not None and positive_after < 0.5:
            warn("very sparse positive signal")

    elif step == "straightening":
        had_signal = qc.get("had_signal", True)
        mask_found = bool(qc.get("source_corridor_mask_found", False))
        corridor_area = _scamp_qc_value(qc, "source_corridor_area_px")
        corridor_total = _scamp_qc_value(qc, "source_corridor_total_intensity")
        after_total = _scamp_qc_value(qc, "straightened_total_intensity")
        corridor_change = _scamp_qc_value(qc, "corridor_intensity_change_percent")
        out_zero = _scamp_qc_value(qc, "output_zero_fraction_percent")
        border0 = _scamp_qc_value(qc, "zero_border_fraction_percent")
        peak_change = _scamp_qc_value(qc, "max_intensity_change_percent_context_only")

        # Hard technical failures only.
        if had_signal is False:
            fail("no signal")
        if after_total is not None and after_total <= 0:
            fail("empty straightened output")
        if out_zero is not None and out_zero > 99.5:
            fail("all-zero straightened output")
        if not mask_found:
            fail("source corridor mask missing")
        if corridor_area is not None and corridor_area <= 0:
            fail("empty source corridor")
        if corridor_total is not None and corridor_total <= 0:
            fail("empty source corridor signal")

        for key in (
            "source_corridor_total_intensity",
            "straightened_total_intensity",
            "corridor_intensity_change_percent",
        ):
            if bad_number(qc.get(key)):
                fail(f"invalid {key}")

        # Human-check warnings: these do not mean the output is unusable.
        if corridor_change is not None:
            if abs(corridor_change) > 25:
                warn("large corridor intensity change")
            elif abs(corridor_change) > 10:
                warn("corridor intensity drift")
            elif abs(corridor_change) > 5:
                warn("minor corridor intensity drift")

        if border0 is not None:
            if border0 > 60:
                warn("large zero border")
            elif border0 > 30:
                warn("zero border")

        if out_zero is not None and out_zero > 85:
            warn("mostly zero output")

        if peak_change is not None and peak_change < -70:
            warn("strong peak smoothing")

    icon = {"ok": "🟢", "warn": "🟡", "fail": "🔴"}.get(level, "🟢")
    color = {"ok": "#7ddc8a", "warn": "#f1c40f", "fail": "#ff6b6b"}.get(level, "#7ddc8a")
    return level, icon, color, reasons


def _scamp_qc_short_text(qc):
    """Compact QC text for preview/sidebar display, with severity icon."""
    if not qc:
        return ""
    qc = _scamp_annotate_qc(qc)
    icon = qc.get("qc_icon", "🟢")
    step = qc.get("step", "QC")

    if step == "background_subtraction":
        est = str(qc.get("background_estimator", "?")).capitalize()
        bg = qc.get("background_slice_mean")
        bg_sig95 = qc.get("background_to_signal95_percent")
        bg_sig99 = qc.get("background_to_signal99_percent")
        retained = qc.get("retained_intensity_percent")
        roi = qc.get("roi_area_px")
        status = {"ok": "PASS", "warn": "CHECK", "fail": "FAIL"}.get(qc.get("qc_level", "ok"), "PASS")
        parts = [f"QC BG {icon} {status}", f"Method: {est}"]
        if roi is not None:
            parts.append(f"ROI: {int(roi)} px")
        if bg is not None:
            parts.append(f"BG: {float(bg):.2f}")
        if bg_sig99 is not None:
            parts.append(f"BG/S99: {float(bg_sig99):.1f}%")
        elif bg_sig95 is not None:
            parts.append(f"BG/S95: {float(bg_sig95):.1f}%")
        if retained is not None:
            parts.append(f"Retained: {float(retained):.0f}%")
        warnings = qc.get("qc_warnings")
        if warnings:
            parts.append(f"⚠ {warnings}")
        return " | ".join(parts)

    if step == "straightening":
        change = qc.get("corridor_intensity_change_percent")
        border0 = qc.get("zero_border_fraction_percent")
        area = qc.get("source_corridor_area_px")
        status = {"ok": "PASS", "warn": "CHECK", "fail": "FAIL"}.get(qc.get("qc_level", "ok"), "PASS")
        parts = [f"QC Straighten {icon} {status}"]
        if change is not None:
            parts.append(f"Corridor Δ: {float(change):+.2f}%")
        if area is not None:
            parts.append(f"Corridor: {int(area)} px")
        if border0 is not None:
            parts.append(f"Border0: {float(border0):.0f}%")
        warnings = qc.get("qc_warnings")
        if warnings:
            parts.append(f"⚠ {warnings}")
        return " | ".join(parts)

    return f"QC {icon} available"


def _scamp_straighten_current_qc(self):
    """Straighten current image and write corridor-aware per-file QC report."""
    ed = self._current_editor()
    if ed is None:
        self._log("No image loaded.")
        return
    if ed.generated_preview:
        self._log(f"{ed.name}: already a straightened preview file.")
        return
    if not self.experiment_dir:
        messagebox.showwarning("No experiment", "No ExperimentID is set.")
        return
    ok = ed.compute_straighten()
    if not ok:
        self._log(f"{ed.name}: FAILED to straighten")
        return

    str_dir = os.path.join(self.experiment_dir, "straightened")
    try:
        os.makedirs(str_dir, exist_ok=True)
        _scamp_qc_dir(self)
    except Exception as exc:
        self._log(f"{ed.name}: could not create output folder: {exc}")
        return

    str_name = build_filename(ed.source_base, self.experiment_id, ed.group, kind="straightened")
    out_path = os.path.join(str_dir, str_name)

    if getattr(ed, "straightened_path", None) and ed.straightened_path != out_path:
        _remove_image_and_preview(ed.straightened_path)

    if getattr(ed, "preview_editor", None) is not None and ed.preview_editor in self.editors:
        self.close_editor(ed.preview_editor)
        ed.preview_editor = None

    try:
        had_signal, _, _ = save_image16(out_path, ed.straightened)
        if not had_signal:
            self._log(f"⚠ {ed.name}: straightened image has NO signal (all zero) — check the source image and landmarks.")
        ed.straightened_path = out_path

        qc = _scamp_straighten_qc_corridor(ed, ed.straightened, ed.name, out_path)
        qc.update({
            "sample": ed.source_base,
            "condition": getattr(ed, "group", ""),
            "had_signal": bool(had_signal),
            "source_path": getattr(ed, "path", ""),
        })
        json_path, csv_path = _scamp_write_qc_report(self, ed.source_base, "straighten", qc)
        qc["qc_json"] = json_path
        qc["qc_csv"] = csv_path
        _scamp_set_editor_qc(ed, qc)

        preview = SampleEditor(self.notebook, out_path, self, generated_preview=True)
        preview.straightened = preview.img.copy()
        preview.qc_summary = qc
        ed.preview_editor = preview
        self.editors.append(preview)
        self.notebook.add(preview, text=preview.name[:24])
        self.notebook.select(preview)
        self._refresh_file_bar()
        self._log(f"{ed.name}: straightened → {str_name}")
        self._log(f"QC straighten report → {json_path}")
    except Exception as exc:
        self._log(f"{ed.name}: straightened, but could not save/open/QC: {exc}")


# ----------------------------------------------------------------------
#  Quantitative display-scale patch for Z-depth normalized projections
# ----------------------------------------------------------------------
# Z-depth normalized projections often have meaningful fractional values
# (for example 0.3..5.0 intensity/µm). Saving those directly to uint16 truncates
# values below 1 to zero, which makes the background look thresholded and can
# create visually black areas after straightening. To keep the requested 16-bit
# TIFF workflow while preserving sub-integer structure, SCAMP stores generated
# Z-depth-normalized projections with a fixed linear scale factor. This does
# not normalize samples against each other; it only changes the storage unit.
SCAMP_ZDEPTH_UINT16_SCALE = 1000.0


def _scamp_apply_zdepth_storage_scale(arr):
    return np.asarray(arr, dtype=np.float64) * float(SCAMP_ZDEPTH_UINT16_SCALE)


def _scamp_subtract_background_qc_scaled_storage(self):
    """Batch background subtraction with scaled uint16 storage for Z-depth output.

    The QC values are computed in native intensity/µm units, but the saved TIFF
    and opened projection use a fixed x1000 storage scale so fractional
    background values are not lost when writing uint16.
    """
    if not self.experiment_dir:
        messagebox.showwarning("No experiment", "No ExperimentID is set.")
        return
    pairs = self._czi_editors_with_rois()
    if not pairs:
        messagebox.showinfo(
            "No CZI + ROI pairs",
            "No open CZI files have an assigned same-name .roi file.\n"
            "Open CZI files, draw a rectangle, then use Tools → Save ROI first."
        )
        return
    if not _HAVE_ROIFILE or not _HAVE_SKDRAW:
        messagebox.showerror(
            "ROI support missing",
            "Background subtraction needs roifile and scikit-image. Update the environment and try again."
        )
        return

    opts = _scamp_ask_background_options(self)
    if opts is None:
        self._log("Background subtraction cancelled.")
        return
    estimator = opts["estimator"]
    percentile = opts["percentile"]

    out_dir = os.path.join(self.experiment_dir, "background_subtracted")
    try:
        os.makedirs(out_dir, exist_ok=True)
        _scamp_qc_dir(self)
    except Exception as exc:
        messagebox.showerror("Output folder error", str(exc))
        return

    fallback_z = None
    created = 0
    skipped = 0
    for ed, roi_path in pairs:
        base = os.path.splitext(os.path.basename(ed.path))[0]
        stack = None
        proj = None
        mask = None
        try:
            self._log(f"Background subtraction: loading {base} ...")
            self.update_idletasks()
            stack, orig_dtype, z_step_meta = load_microscopy_stack(ed.path)
            history = list(getattr(ed, "transform_history", []))
            if history:
                self._log(f"Background subtraction: applying {len(history)} geometry operation(s) for {base} ...")
                self.update_idletasks()
                stack = apply_stack_transform_history(stack, history)
            z_step = getattr(ed, "z_step_um", None) or z_step_meta
            if z_step is None or z_step <= 0:
                if fallback_z is None:
                    fallback_z = self._ask_fallback_z_step()
                    if fallback_z is None:
                        self._log(f"Skipped {base}: no Z-step value available.")
                        skipped += 1
                        continue
                z_step = fallback_z

            roi_rect = getattr(ed, "background_roi_rect", None)
            if roi_rect is not None:
                mask = rectangle_mask_from_rect(roi_rect, stack.shape[1:])
            else:
                mask = load_roi_mask_from_file(roi_path, stack.shape[1:])

            proj_native, qc = project_stack_with_background_mask_qc(
                stack, mask, method="zdepth", z_step_um=z_step,
                estimator=estimator, percentile=percentile)
            proj = _scamp_apply_zdepth_storage_scale(proj_native)
            qc.update({
                "sample": base,
                "source_path": ed.path,
                "roi_path": roi_path,
                "condition": getattr(ed, "group", ""),
                "geometry_operations": len(history),
                "storage_scale_factor": SCAMP_ZDEPTH_UINT16_SCALE,
                "storage_unit_note": (
                    "Saved TIFF pixel values are native Z-depth-normalized intensity multiplied "
                    f"by {SCAMP_ZDEPTH_UINT16_SCALE:g}. This preserves sub-integer background "
                    "structure in uint16 TIFF output and avoids threshold-like black background."
                ),
            })

            out_name = build_filename(
                base,
                self.experiment_id,
                ed.group,
                kind="background_subtracted_Zdepth_normalised_sumIP",
                ext=".tif",
            )
            out_path = os.path.join(out_dir, out_name)
            had_signal, _, _ = save_image16(out_path, proj)
            qc["output_path"] = out_path
            qc["had_signal"] = bool(had_signal)
            json_path, csv_path = _scamp_write_qc_report(self, base, "background", qc)
            qc["qc_json"] = json_path
            qc["qc_csv"] = csv_path

            if not had_signal:
                self._log(f"⚠ {ed.name}: background-subtracted projection has no signal. Check the ROI.")

            proj_ed = SampleEditor(
                self.notebook,
                out_path,
                self,
                image_array=proj,
                name=out_name,
                group=ed.group,
            )
            proj_ed.source_base = base
            proj_ed.normalized = proj
            proj_ed.storage_scale_factor = float(qc.get("storage_scale_factor") or SCAMP_ZDEPTH_UINT16_SCALE)
            proj_ed.storage_unit_note = qc.get("storage_unit_note", "")
            _scamp_set_editor_qc(proj_ed, qc)
            _scamp_set_editor_qc(ed, qc)
            self.editors.append(proj_ed)
            self.notebook.add(proj_ed, text=proj_ed.name[:24])
            self.notebook.select(proj_ed)
            created += 1
            self._log(f"Background-subtracted projection → {out_name}")
            self._log(f"Applied fixed uint16 storage scale ×{SCAMP_ZDEPTH_UINT16_SCALE:g} to preserve fractional intensity/background.")
            self._log(f"QC background report → {json_path}")
        except Exception as exc:
            skipped += 1
            self._log(f"Skipped {base}: background subtraction failed: {exc}")
        finally:
            for varname in ("stack", "mask", "proj"):
                try:
                    del locals()[varname]
                except Exception:
                    pass
            gc.collect()
            self.update_idletasks()

    self._refresh_file_bar()
    messagebox.showinfo(
        "Background subtraction done",
        f"Created {created} projection(s).\nSkipped {skipped}.\nOutput folder:\n{out_dir}\n\n"
        f"Z-depth TIFF storage scale: ×{SCAMP_ZDEPTH_UINT16_SCALE:g}\n"
        f"QC reports:\n{_scamp_qc_dir(self)}",
    )


# ----------------------------------------------------------------------
#  Save/reload storage QC patch
# ----------------------------------------------------------------------
def _scamp_positive_fraction(arr):
    a = np.asarray(arr, dtype=np.float64)
    return float(np.count_nonzero(a > 0) / float(a.size) * 100.0) if a.size else None


def _scamp_fraction_between_zero_and_one(arr):
    a = np.asarray(arr, dtype=np.float64)
    return float(np.count_nonzero((a > 0) & (a < 1)) / float(a.size) * 100.0) if a.size else None


def _scamp_saturation_fraction_uint16_range(arr):
    a = np.asarray(arr, dtype=np.float64)
    return float(np.count_nonzero(a >= 65535.0) / float(a.size) * 100.0) if a.size else None


def _scamp_saved_image_roundtrip_qc(arr_before_save, path, prefix="saved_image"):
    """Compare the in-memory array to the TIFF as reloaded from disk.

    This detects whether the quantitative image changed during uint16 storage,
    for example because many meaningful sub-1 values were truncated to zero or
    because bright values saturated at 65535.
    """
    qc = {}
    before = np.asarray(arr_before_save, dtype=np.float64)
    qc[f"{prefix}_pre_save_total_intensity"] = _scamp_float(np.sum(before, dtype=np.float64)) if before.size else None
    qc[f"{prefix}_pre_save_mean_intensity"] = _scamp_float(np.mean(before)) if before.size else None
    qc[f"{prefix}_pre_save_max_intensity"] = _scamp_float(np.max(before)) if before.size else None
    qc[f"{prefix}_pre_save_positive_fraction_percent"] = _scamp_float(_scamp_positive_fraction(before))
    qc[f"{prefix}_pre_save_positive_below_one_fraction_percent"] = _scamp_float(_scamp_fraction_between_zero_and_one(before))
    qc[f"{prefix}_pre_save_saturation_fraction_percent"] = _scamp_float(_scamp_saturation_fraction_uint16_range(before))
    try:
        reloaded, reloaded_dtype = load_image_any(path)
        reloaded = np.asarray(reloaded, dtype=np.float64)
        qc[f"{prefix}_roundtrip_ok"] = bool(reloaded.shape == before.shape)
        qc[f"{prefix}_reloaded_dtype"] = str(reloaded_dtype)
        qc[f"{prefix}_reloaded_total_intensity"] = _scamp_float(np.sum(reloaded, dtype=np.float64)) if reloaded.size else None
        qc[f"{prefix}_reloaded_mean_intensity"] = _scamp_float(np.mean(reloaded)) if reloaded.size else None
        qc[f"{prefix}_reloaded_max_intensity"] = _scamp_float(np.max(reloaded)) if reloaded.size else None
        qc[f"{prefix}_reloaded_positive_fraction_percent"] = _scamp_float(_scamp_positive_fraction(reloaded))
        qc[f"{prefix}_total_intensity_change_after_reload_percent"] = _scamp_float(_scamp_percent_change(qc[f"{prefix}_pre_save_total_intensity"], qc[f"{prefix}_reloaded_total_intensity"]))
        pre_pos = qc[f"{prefix}_pre_save_positive_fraction_percent"]
        rel_pos = qc[f"{prefix}_reloaded_positive_fraction_percent"]
        if pre_pos is not None and pre_pos > 0 and rel_pos is not None:
            qc[f"{prefix}_positive_pixel_retention_percent"] = _scamp_float(rel_pos / pre_pos * 100.0)
        else:
            qc[f"{prefix}_positive_pixel_retention_percent"] = None
        qc[f"{prefix}_storage_note"] = (
            "Compares the in-memory quantitative image just before saving with "
            "the uint16 TIFF reloaded from disk. Large positive-pixel loss or "
            "large total-intensity change indicates storage quantization/clipping, "
            "not a biological or straightening effect."
        )
    except Exception as exc:
        qc[f"{prefix}_roundtrip_ok"] = False
        qc[f"{prefix}_roundtrip_error"] = str(exc)
    return qc


def _scamp_add_native_projection_storage_risk_qc(qc, proj_native):
    """Add native Z-depth projection storage-risk fields before scaling."""
    a = np.asarray(proj_native, dtype=np.float64)
    qc.update({
        "native_projection_total_intensity_before_storage_scale": _scamp_float(np.sum(a, dtype=np.float64)) if a.size else None,
        "native_projection_mean_intensity_before_storage_scale": _scamp_float(np.mean(a)) if a.size else None,
        "native_projection_max_intensity_before_storage_scale": _scamp_float(np.max(a)) if a.size else None,
        "native_projection_positive_fraction_percent": _scamp_float(_scamp_positive_fraction(a)),
        "native_projection_positive_below_one_fraction_percent": _scamp_float(_scamp_fraction_between_zero_and_one(a)),
        "native_projection_storage_risk_note": (
            "For native Z-depth-normalised projections, positive values below 1 "
            "would be truncated to 0 by direct uint16 saving. SCAMP stores these "
            "projections with a fixed linear scale factor to preserve sub-integer structure."
        ),
    })


def _scamp_subtract_background_qc_scaled_storage_roundtrip(self):
    """Batch background subtraction with estimator dialog and storage roundtrip QC."""
    if not self.experiment_dir:
        messagebox.showwarning("No experiment", "No ExperimentID is set.")
        return
    pairs = self._czi_editors_with_rois()
    if not pairs:
        messagebox.showinfo(
            "No CZI + ROI pairs",
            "No open CZI files have an assigned same-name .roi file.\n"
            "Open CZI files, draw a rectangle, then use Tools → Save ROI first."
        )
        return
    if not _HAVE_ROIFILE or not _HAVE_SKDRAW:
        messagebox.showerror(
            "ROI support missing",
            "Background subtraction needs roifile and scikit-image. Update the environment and try again."
        )
        return

    opts = _scamp_ask_background_options(self)
    if opts is None:
        self._log("Background subtraction cancelled.")
        return
    estimator = opts["estimator"]
    percentile = opts["percentile"]

    out_dir = os.path.join(self.experiment_dir, "background_subtracted")
    try:
        os.makedirs(out_dir, exist_ok=True)
        _scamp_qc_dir(self)
    except Exception as exc:
        messagebox.showerror("Output folder error", str(exc))
        return

    fallback_z = None
    created = 0
    skipped = 0
    for ed, roi_path in pairs:
        base = os.path.splitext(os.path.basename(ed.path))[0]
        stack = None
        proj = None
        mask = None
        try:
            self._log(f"Background subtraction: loading {base} ...")
            self.update_idletasks()
            stack, orig_dtype, z_step_meta = load_microscopy_stack(ed.path)
            history = list(getattr(ed, "transform_history", []))
            if history:
                self._log(f"Background subtraction: applying {len(history)} geometry operation(s) for {base} ...")
                self.update_idletasks()
                stack = apply_stack_transform_history(stack, history)
            z_step = getattr(ed, "z_step_um", None) or z_step_meta
            if z_step is None or z_step <= 0:
                if fallback_z is None:
                    fallback_z = self._ask_fallback_z_step()
                    if fallback_z is None:
                        self._log(f"Skipped {base}: no Z-step value available.")
                        skipped += 1
                        continue
                z_step = fallback_z

            roi_rect = getattr(ed, "background_roi_rect", None)
            if roi_rect is not None:
                mask = rectangle_mask_from_rect(roi_rect, stack.shape[1:])
            else:
                mask = load_roi_mask_from_file(roi_path, stack.shape[1:])

            proj_native, qc = project_stack_with_background_mask_qc(
                stack, mask, method="zdepth", z_step_um=z_step,
                estimator=estimator, percentile=percentile)
            _scamp_add_native_projection_storage_risk_qc(qc, proj_native)
            proj = _scamp_apply_zdepth_storage_scale(proj_native)
            qc.update({
                "sample": base,
                "source_path": ed.path,
                "roi_path": roi_path,
                "condition": getattr(ed, "group", ""),
                "geometry_operations": len(history),
                "storage_scale_factor": SCAMP_ZDEPTH_UINT16_SCALE,
                "storage_unit_note": (
                    "Saved TIFF pixel values are native Z-depth-normalized intensity multiplied "
                    f"by {SCAMP_ZDEPTH_UINT16_SCALE:g}. This preserves sub-integer background "
                    "structure in uint16 TIFF output and avoids threshold-like black background."
                ),
            })

            out_name = build_filename(
                base,
                self.experiment_id,
                ed.group,
                kind="background_subtracted_Zdepth_normalised_sumIP",
                ext=".tif",
            )
            out_path = os.path.join(out_dir, out_name)
            had_signal, _, _ = save_image16(out_path, proj)
            qc["output_path"] = out_path
            qc["had_signal"] = bool(had_signal)
            qc.update(_scamp_saved_image_roundtrip_qc(proj, out_path, prefix="saved_projection"))
            json_path, csv_path = _scamp_write_qc_report(self, base, "background", qc)
            qc["qc_json"] = json_path
            qc["qc_csv"] = csv_path

            if not had_signal:
                self._log(f"⚠ {ed.name}: background-subtracted projection has no signal. Check the ROI.")

            proj_ed = SampleEditor(
                self.notebook,
                out_path,
                self,
                image_array=proj,
                name=out_name,
                group=ed.group,
            )
            proj_ed.source_base = base
            proj_ed.normalized = proj
            proj_ed.storage_scale_factor = float(qc.get("storage_scale_factor") or SCAMP_ZDEPTH_UINT16_SCALE)
            proj_ed.storage_unit_note = qc.get("storage_unit_note", "")
            _scamp_set_editor_qc(proj_ed, qc)
            _scamp_set_editor_qc(ed, qc)
            self.editors.append(proj_ed)
            self.notebook.add(proj_ed, text=proj_ed.name[:24])
            self.notebook.select(proj_ed)
            created += 1
            self._log(f"Background-subtracted projection → {out_name}")
            self._log(f"Applied fixed uint16 storage scale ×{SCAMP_ZDEPTH_UINT16_SCALE:g} to preserve fractional intensity/background.")
            self._log(f"QC storage roundtrip: positive retention = {qc.get('saved_projection_positive_pixel_retention_percent')}%")
            self._log(f"QC background report → {json_path}")
        except Exception as exc:
            skipped += 1
            self._log(f"Skipped {base}: background subtraction failed: {exc}")
        finally:
            for varname in ("stack", "mask", "proj"):
                try:
                    del locals()[varname]
                except Exception:
                    pass
            gc.collect()
            self.update_idletasks()

    self._refresh_file_bar()
    messagebox.showinfo(
        "Background subtraction done",
        f"Created {created} projection(s).\nSkipped {skipped}.\nOutput folder:\n{out_dir}\n\n"
        f"Z-depth TIFF storage scale: ×{SCAMP_ZDEPTH_UINT16_SCALE:g}\n"
        f"QC reports:\n{_scamp_qc_dir(self)}",
    )


def _scamp_straighten_current_qc_roundtrip(self):
    """Straighten current image and write corridor-aware + save/reload QC."""
    ed = self._current_editor()
    if ed is None:
        self._log("No image loaded.")
        return
    if ed.generated_preview:
        self._log(f"{ed.name}: already a straightened preview file.")
        return
    if not self.experiment_dir:
        messagebox.showwarning("No experiment", "No ExperimentID is set.")
        return
    ok = ed.compute_straighten()
    if not ok:
        self._log(f"{ed.name}: FAILED to straighten")
        return

    str_dir = os.path.join(self.experiment_dir, "straightened")
    try:
        os.makedirs(str_dir, exist_ok=True)
        _scamp_qc_dir(self)
    except Exception as exc:
        self._log(f"{ed.name}: could not create output folder: {exc}")
        return

    str_name = build_filename(ed.source_base, self.experiment_id, ed.group, kind="straightened")
    out_path = os.path.join(str_dir, str_name)

    if getattr(ed, "straightened_path", None) and ed.straightened_path != out_path:
        _remove_image_and_preview(ed.straightened_path)

    if getattr(ed, "preview_editor", None) is not None and ed.preview_editor in self.editors:
        self.close_editor(ed.preview_editor)
        ed.preview_editor = None

    try:
        had_signal, _, _ = save_image16(out_path, ed.straightened)
        if not had_signal:
            self._log(f"⚠ {ed.name}: straightened image has NO signal (all zero) — check the source image and landmarks.")
        ed.straightened_path = out_path

        qc = _scamp_straighten_qc_corridor(ed, ed.straightened, ed.name, out_path)
        qc.update({
            "sample": ed.source_base,
            "condition": getattr(ed, "group", ""),
            "had_signal": bool(had_signal),
            "source_path": getattr(ed, "path", ""),
        })
        qc.update(_scamp_saved_image_roundtrip_qc(ed.straightened, out_path, prefix="saved_straightened"))
        json_path, csv_path = _scamp_write_qc_report(self, ed.source_base, "straighten", qc)
        qc["qc_json"] = json_path
        qc["qc_csv"] = csv_path
        _scamp_set_editor_qc(ed, qc)

        preview = SampleEditor(self.notebook, out_path, self, generated_preview=True)
        preview.straightened = preview.img.copy()
        preview.qc_summary = qc
        _scamp_set_editor_qc(preview, qc)
        ed.preview_editor = preview
        self.editors.append(preview)
        self.notebook.add(preview, text=preview.name[:24])
        self.notebook.select(preview)
        self._refresh_file_bar()
        self._log(f"{ed.name}: straightened → {str_name}")
        self._log(f"QC storage roundtrip: positive retention = {qc.get('saved_straightened_positive_pixel_retention_percent')}%")
        self._log(f"QC straighten report → {json_path}")
    except Exception as exc:
        self._log(f"{ed.name}: straightened, but could not save/open/QC: {exc}")


# Reinstall final background handler last.
App.subtract_background = _scamp_subtract_background_qc_scaled_storage_roundtrip


# ----------------------------------------------------------------------
#  Final storage-preserving straightening patch
# ----------------------------------------------------------------------
def _scamp_fraction_positive_below_one(arr):
    a = np.asarray(arr, dtype=np.float64)
    return float(np.count_nonzero((a > 0) & (a < 1)) / float(a.size) * 100.0) if a.size else 0.0


def _scamp_storage_array_for_uint16(arr, input_storage_scale=1.0):
    """Return (array_to_save, save_scale, note) for uint16-safe storage.

    ``input_storage_scale`` is metadata describing the fixed SCAMP storage-unit
    scale that this product should use. It is NOT proof that the in-memory array
    is already scaled. Straightened images often carry the projection's storage
    metadata while the actual warped array still contains native float values.

    If many positive pixels are between 0 and 1, saving directly to uint16 would
    truncate them to zero. In that case, convert to the fixed storage unit before
    writing. The multiplier is global, not per-image normalization.
    """
    a = np.asarray(arr, dtype=np.float64)
    existing_scale = float(input_storage_scale or 1.0)
    if existing_scale <= 0 or not np.isfinite(existing_scale):
        existing_scale = 1.0

    below_one = _scamp_fraction_positive_below_one(a)
    maxv = float(np.nanmax(a)) if a.size else 0.0

    if below_one > 1.0:
        if existing_scale > 1.0:
            scale = existing_scale
            sat_note = ""
            if maxv * scale >= 65535.0:
                sat_note = " Some values may saturate; check saturation_fraction in QC."
            return a * scale, 1.0, (
                f"Converted native float array into existing SCAMP storage units "
                f"(×{existing_scale:g}) before uint16 TIFF saving. This preserves "
                "sub-1 quantitative values and does not normalize between samples." + sat_note
            )

        scale = float(SCAMP_ZDEPTH_UINT16_SCALE)
        sat_note = ""
        if maxv * scale >= 65535.0:
            sat_note = " Some values may saturate; check saturation_fraction in QC."
        return a * scale, scale, (
            f"Applied fixed ×{scale:g} storage scale before uint16 TIFF saving "
            "to preserve sub-1 quantitative values. This is storage scaling, "
            "not biological normalization." + sat_note
        )

    return a, 1.0, "No storage scaling needed; positive sub-1 fraction is low."


def _scamp_straighten_current_qc_roundtrip_storage_safe(self):
    """Straighten current image, QC native corridor intensity, save uint16 safely."""
    ed = self._current_editor()
    if ed is None:
        self._log("No image loaded.")
        return
    if ed.generated_preview:
        self._log(f"{ed.name}: already a straightened preview file.")
        return
    if not self.experiment_dir:
        messagebox.showwarning("No experiment", "No ExperimentID is set.")
        return
    ok = ed.compute_straighten()
    if not ok:
        self._log(f"{ed.name}: FAILED to straighten")
        return

    str_dir = os.path.join(self.experiment_dir, "straightened")
    try:
        os.makedirs(str_dir, exist_ok=True)
        _scamp_qc_dir(self)
    except Exception as exc:
        self._log(f"{ed.name}: could not create output folder: {exc}")
        return

    str_name = build_filename(ed.source_base, self.experiment_id, ed.group, kind="straightened")
    out_path = os.path.join(str_dir, str_name)

    if getattr(ed, "straightened_path", None) and ed.straightened_path != out_path:
        _remove_image_and_preview(ed.straightened_path)

    if getattr(ed, "preview_editor", None) is not None and ed.preview_editor in self.editors:
        self.close_editor(ed.preview_editor)
        ed.preview_editor = None

    try:
        # QC is computed on the native in-memory straightened image, before any
        # storage-only scaling is applied.
        qc = _scamp_straighten_qc_corridor(ed, ed.straightened, ed.name, out_path)
        qc.update({
            "sample": ed.source_base,
            "condition": getattr(ed, "group", ""),
            "source_path": getattr(ed, "path", ""),
        })

        input_scale = float(getattr(ed, "storage_scale_factor", 1.0) or 1.0)
        save_arr, save_scale, save_note = _scamp_storage_array_for_uint16(
            ed.straightened,
            input_storage_scale=input_scale,
        )
        qc.update({
            "straightened_input_storage_scale_factor": input_scale,
            "straightened_save_storage_scale_factor": save_scale,
            "straightened_effective_storage_scale_factor": input_scale * save_scale,
            "straightened_storage_note": save_note,
            "straightened_native_positive_below_one_fraction_percent": _scamp_float(_scamp_fraction_positive_below_one(ed.straightened)),
        })

        had_signal, _, _ = save_image16(out_path, save_arr)
        if not had_signal:
            self._log(f"⚠ {ed.name}: straightened image has NO signal (all zero) — check the source image and landmarks.")
        ed.straightened_path = out_path
        ed.straightened_storage_scale_factor = input_scale * save_scale

        qc["had_signal"] = bool(had_signal)
        qc.update(_scamp_saved_image_roundtrip_qc(save_arr, out_path, prefix="saved_straightened_storage_units"))

        # Add a native-unit estimate of what was reloaded, when possible.
        eff_scale = input_scale * save_scale
        if eff_scale and eff_scale > 0:
            rel_total = qc.get("saved_straightened_storage_units_reloaded_total_intensity")
            rel_mean = qc.get("saved_straightened_storage_units_reloaded_mean_intensity")
            rel_max = qc.get("saved_straightened_storage_units_reloaded_max_intensity")
            qc["saved_straightened_reloaded_total_intensity_native_estimate"] = _scamp_float(rel_total / eff_scale) if rel_total is not None else None
            qc["saved_straightened_reloaded_mean_intensity_native_estimate"] = _scamp_float(rel_mean / eff_scale) if rel_mean is not None else None
            qc["saved_straightened_reloaded_max_intensity_native_estimate"] = _scamp_float(rel_max / eff_scale) if rel_max is not None else None
            qc["saved_straightened_native_total_change_after_reload_percent_estimate"] = _scamp_float(
                _scamp_percent_change(qc.get("straightened_total_intensity"), qc.get("saved_straightened_reloaded_total_intensity_native_estimate"))
            )

        # Storage quantization should be visible as CHECK, not as a biological
        # straightening fail. Keep FAIL reserved for technical failures.
        storage_ret = qc.get("saved_straightened_storage_units_positive_pixel_retention_percent")
        native_change = qc.get("saved_straightened_native_total_change_after_reload_percent_estimate")
        extra_checks = []
        if storage_ret is not None and storage_ret < 95.0:
            extra_checks.append("storage positive-pixel loss")
        if native_change is not None and abs(native_change) > 5.0:
            extra_checks.append("storage total-intensity drift")
        if extra_checks and qc.get("qc_status") == "PASS":
            qc["qc_status"] = "CHECK"
            qc["qc_level"] = "warn"
            qc["qc_icon"] = "🟡"
        if extra_checks:
            warnings = list(qc.get("qc_warnings") or []) if isinstance(qc.get("qc_warnings"), list) else []
            warnings.extend(extra_checks)
            qc["qc_warnings"] = "; ".join(dict.fromkeys(str(w) for w in warnings if w))

        json_path, csv_path = _scamp_write_qc_report(self, ed.source_base, "straighten", qc)
        qc["qc_json"] = json_path
        qc["qc_csv"] = csv_path
        _scamp_set_editor_qc(ed, qc)

        preview = SampleEditor(self.notebook, out_path, self, generated_preview=True, image_array=save_arr)
        preview.straightened = save_arr.copy()
        preview.qc_summary = qc
        preview.storage_scale_factor = input_scale * save_scale
        _scamp_set_editor_qc(preview, qc)
        ed.preview_editor = preview
        self.editors.append(preview)
        self.notebook.add(preview, text=preview.name[:24])
        self.notebook.select(preview)
        self._refresh_file_bar()
        self._log(f"{ed.name}: straightened → {str_name}")
        self._log(f"Straightened storage scale: ×{input_scale * save_scale:g}. {save_note}")
        self._log(f"QC storage roundtrip: positive retention = {qc.get('saved_straightened_storage_units_positive_pixel_retention_percent')}%")
        self._log(f"QC straighten report → {json_path}")
    except Exception as exc:
        self._log(f"{ed.name}: straightened, but could not save/open/QC: {exc}")

# Final handler override: corridor QC + storage-safe uint16 straightened output.
App.straighten_current = _scamp_straighten_current_qc_roundtrip_storage_safe


# ======================================================================
#  Final uniform storage-unit policy patch
# ======================================================================
def _scamp_editor_storage_scale(ed):
    """Return the quantitative storage scale attached to an editor."""
    try:
        return float(getattr(ed, "storage_scale_factor", 1.0) or 1.0)
    except Exception:
        return 1.0


def _scamp_mark_editor_storage(ed, scale, note=None):
    """Attach SCAMP storage-scale metadata to an editor object."""
    try:
        ed.storage_scale_factor = float(scale or 1.0)
        if note is not None:
            ed.storage_unit_note = str(note)
    except Exception:
        pass


def _scamp_scaled_for_storage_and_analysis(arr, input_scale=1.0):
    """Return an array in the storage units used for saving and analysis.

    SCAMP analyses generated TIFF products in their saved storage units. For
    Z-depth-normalized data this is intentionally a fixed global multiplier
    (SCAMP_ZDEPTH_UINT16_SCALE) so sub-integer native intensities are preserved
    in uint16 TIFF. This is not per-sample normalization: the same multiplier is
    used for every generated Z-depth product.
    """
    return _scamp_storage_array_for_uint16(arr, input_storage_scale=input_scale)


def _scamp_process_all_storage_consistent(self):
    """Process straightened images using the same storage-unit convention.

    This avoids mixing native in-memory float images with scaled TIFFs from
    disk. If a source editor is straightened on demand, it is converted to the
    same storage units before saving and before profile extraction.
    """
    if not getattr(self, "editors", None):
        messagebox.showinfo("Nothing to do", "Load some images first.")
        return
    if not self.experiment_dir:
        messagebox.showwarning("No experiment", "No ExperimentID is set.")
        return

    out_dir = self.experiment_dir
    str_dir = os.path.join(out_dir, "straightened")
    os.makedirs(str_dir, exist_ok=True)

    profiles = []
    seen = set()

    def add_profile(name, group, img, origin):
        key = os.path.abspath(origin) if origin else name
        if key in seen:
            return False
        try:
            profile, start = _scamp_profile_from_image_array(img)
        except Exception as exc:
            self._log(f"{name}: profile extraction failed: {exc}")
            return False
        if len(profile) == 0 or not np.any(np.isfinite(profile)):
            self._log(f"{name}: empty profile — skipped.")
            return False
        profiles.append((name, group or "", profile))
        seen.add(key)
        grp = group if group else "(no condition)"
        self._log(f"{name}: full profile length {len(profile)} (condition: {grp}; no trimming).")
        return True

    for ed in list(self.editors):
        if getattr(ed, "is_stack_editor", False) or getattr(ed, "is_deferred_czi", False):
            continue

        if getattr(ed, "generated_preview", False):
            img = getattr(ed, "img", None)
            if img is not None:
                add_profile(ed.name, getattr(ed, "group", ""), img, getattr(ed, "path", ed.name))
            continue

        if getattr(ed, "straightened", None) is None:
            try:
                ed.compute_straighten()
            except Exception as exc:
                self._log(f"{ed.name}: skipped (could not straighten: {exc}).")
                continue
        if getattr(ed, "straightened", None) is None:
            self._log(f"{ed.name}: skipped (could not straighten).")
            continue

        str_name = build_filename(ed.source_base, self.experiment_id, getattr(ed, "group", ""), kind="straightened")
        out_path = os.path.join(str_dir, str_name)
        input_scale = _scamp_editor_storage_scale(ed)
        save_arr, save_scale, save_note = _scamp_scaled_for_storage_and_analysis(ed.straightened, input_scale=input_scale)
        effective_scale = input_scale * save_scale
        try:
            if getattr(ed, "straightened_path", None) != out_path:
                save_image16(out_path, save_arr)
                ed.straightened_path = out_path
            ed.straightened = save_arr
            _scamp_mark_editor_storage(ed, effective_scale, save_note)
        except Exception as exc:
            self._log(f"{ed.name}: could not save straightened image: {exc}")
        add_profile(ed.name, getattr(ed, "group", ""), save_arr, out_path)

    # Disk fallback: saved straightened TIFFs are already in storage units.
    try:
        disk_files = []
        for fn in os.listdir(str_dir):
            low = fn.lower()
            if low.endswith((".tif", ".tiff")) and "_straightened" in low and "_preview" not in low:
                disk_files.append(os.path.join(str_dir, fn))
        for path in sorted(disk_files):
            key = os.path.abspath(path)
            if key in seen:
                continue
            try:
                img, _dtype = load_image_any(path)
            except Exception as exc:
                self._log(f"{os.path.basename(path)}: could not read straightened TIFF: {exc}")
                continue
            group = _scamp_group_from_straightened_name(path, self.experiment_id)
            add_profile(os.path.basename(path), group, img, path)
    except Exception as exc:
        self._log(f"Could not scan straightened folder: {exc}")

    if not profiles:
        messagebox.showwarning(
            "No profiles",
            "No usable profiles were produced. Check that straightened TIFFs are readable."
        )
        self._log("No usable profiles were produced after checking memory and the straightened/ folder.")
        return

    names, groups, matrix = assemble_matrix(profiles)

    opts = self._pick_heatmap_options()
    n_groups = 0
    if opts is None:
        self._log("Heatmap cancelled; tables will still be written.")
    else:
        try:
            fig, n_groups = self._build_cohort_heatmap_figure(names, groups, matrix, cmap_spec=opts["cmap"])
            base = build_filename("COHORT_HEATMAP", self.experiment_id, "", ext="")
            if opts.get("pdf"):
                out_pdf = os.path.join(out_dir, base + ".pdf")
                fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
                self._log(f"Wrote {out_pdf}")
            if opts.get("png"):
                out_png = os.path.join(out_dir, base + ".png")
                fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
                self._log(f"Wrote {out_png}")
            self._show_heatmap_window(fig, title=f"Cohort heatmap — {self.experiment_id}")
        except Exception as exc:
            self._log(f"Cohort heatmap failed: {exc}")

    try:
        written = write_tables(out_dir, self.experiment_id, names, groups, matrix)
        for w in written:
            self._log(f"Wrote {w}")
    except Exception as exc:
        self._log(f"Table export failed: {exc}")

    cond_note = (f"\n{n_groups} conditions color-coded on the heatmap." if n_groups and n_groups > 1 else "")
    messagebox.showinfo(
        "Done",
        f"Processed {len(profiles)} usable straightened sample(s).\n"
        f"Output in:\n{out_dir}{cond_note}\n\n"
        f"Analysis used saved SCAMP storage units. For Z-depth products this is native intensity ×{SCAMP_ZDEPTH_UINT16_SCALE:g}."
    )
    self._log(f"FINISHED: {len(profiles)} profiles → {out_dir}")
    self._log(f"Storage-unit policy: generated Z-depth TIFFs use fixed ×{SCAMP_ZDEPTH_UINT16_SCALE:g}; no per-sample scaling.")


App.process_all = _scamp_process_all_storage_consistent


# ----------------------------------------------------------------------
# Final Process all native-unit output patch
# ----------------------------------------------------------------------
def _scamp_process_all_native_units(self):
    """Process straightened images in native quantitative units.

    Generated TIFFs are stored as uint16 SCAMP storage units, usually native
    Z-depth-normalised intensity × SCAMP_ZDEPTH_UINT16_SCALE. This is necessary
    to prevent sub-1 float values from being truncated to zero during TIFF
    saving. Downstream CSV/XLSX/heatmap analysis, however, should report native
    quantitative values. Therefore Process all divides storage-unit images by
    the relevant fixed storage scale before extracting full-length column-sum profiles.
    """
    if not getattr(self, "editors", None):
        messagebox.showinfo("Nothing to do", "Load some images first.")
        return
    if not self.experiment_dir:
        messagebox.showwarning("No experiment", "No ExperimentID is set.")
        return

    out_dir = self.experiment_dir
    str_dir = os.path.join(out_dir, "straightened")
    os.makedirs(str_dir, exist_ok=True)

    profiles = []
    seen = set()

    def _native_from_storage(img, scale):
        try:
            scale = float(scale or 1.0)
        except Exception:
            scale = 1.0
        if scale <= 0 or not np.isfinite(scale):
            scale = 1.0
        arr = np.asarray(img, dtype=np.float64)
        return arr / scale if scale != 1.0 else arr

    def add_profile(name, group, img, origin, storage_scale=1.0):
        key = os.path.abspath(origin) if origin else name
        if key in seen:
            return False
        try:
            img_native = _native_from_storage(img, storage_scale)
            profile, start = _scamp_profile_from_image_array(img_native)
        except Exception as exc:
            self._log(f"{name}: profile extraction failed: {exc}")
            return False
        if len(profile) == 0 or not np.any(np.isfinite(profile)):
            self._log(f"{name}: empty profile — skipped.")
            return False
        profiles.append((name, group or "", profile))
        seen.add(key)
        grp = group if group else "(no condition)"
        if storage_scale and float(storage_scale or 1.0) != 1.0:
            self._log(
                f"{name}: full profile length {len(profile)} (condition: {grp}; no trimming; "
                f"native units after /{float(storage_scale):g} storage-scale correction)."
            )
        else:
            self._log(f"{name}: full profile length {len(profile)} (condition: {grp}; no trimming).")
        return True

    # 1) In-memory editors.
    for ed in list(self.editors):
        if getattr(ed, "is_stack_editor", False) or getattr(ed, "is_deferred_czi", False):
            continue

        if getattr(ed, "generated_preview", False):
            img = getattr(ed, "img", None)
            if img is not None:
                scale = _scamp_editor_storage_scale(ed)
                add_profile(ed.name, getattr(ed, "group", ""), img, getattr(ed, "path", ed.name), storage_scale=scale)
            continue

        if getattr(ed, "straightened", None) is None:
            try:
                ed.compute_straighten()
            except Exception as exc:
                self._log(f"{ed.name}: skipped (could not straighten: {exc}).")
                continue
        if getattr(ed, "straightened", None) is None:
            self._log(f"{ed.name}: skipped (could not straighten).")
            continue

        str_name = build_filename(ed.source_base, self.experiment_id, getattr(ed, "group", ""), kind="straightened")
        out_path = os.path.join(str_dir, str_name)
        input_scale = _scamp_editor_storage_scale(ed)
        save_arr, save_scale, save_note = _scamp_scaled_for_storage_and_analysis(ed.straightened, input_scale=input_scale)
        effective_scale = input_scale * save_scale
        try:
            if getattr(ed, "straightened_path", None) != out_path:
                save_image16(out_path, save_arr)
                ed.straightened_path = out_path
            # Keep the editor aligned with the saved file, but always divide by
            # effective_scale before full-length profile extraction below.
            ed.straightened = save_arr
            _scamp_mark_editor_storage(ed, effective_scale, save_note)
        except Exception as exc:
            self._log(f"{ed.name}: could not save straightened image: {exc}")
        add_profile(ed.name, getattr(ed, "group", ""), save_arr, out_path, storage_scale=effective_scale)

    # 2) Disk fallback: saved generated straightened TIFFs are in fixed SCAMP
    # storage units. Convert them back to native units before analysis.
    try:
        disk_files = []
        for fn in os.listdir(str_dir):
            low = fn.lower()
            if low.endswith((".tif", ".tiff")) and "_straightened" in low and "_preview" not in low:
                disk_files.append(os.path.join(str_dir, fn))
        for path in sorted(disk_files):
            key = os.path.abspath(path)
            if key in seen:
                continue
            try:
                img, _dtype = load_image_any(path)
            except Exception as exc:
                self._log(f"{os.path.basename(path)}: could not read straightened TIFF: {exc}")
                continue
            group = _scamp_group_from_straightened_name(path, self.experiment_id)
            # SCAMP-generated straightened TIFFs inherit the global Z-depth
            # storage scale. If a future non-scaled image enters this folder,
            # this can be changed to read sidecar metadata; current SCAMP
            # outputs use this fixed global policy.
            add_profile(os.path.basename(path), group, img, path, storage_scale=SCAMP_ZDEPTH_UINT16_SCALE)
    except Exception as exc:
        self._log(f"Could not scan straightened folder: {exc}")

    if not profiles:
        messagebox.showwarning(
            "No profiles",
            "No usable profiles were produced. Check that straightened TIFFs are readable."
        )
        self._log("No usable profiles were produced after checking memory and the straightened/ folder.")
        return

    names, groups, matrix = assemble_matrix(profiles)

    opts = self._pick_heatmap_options()
    n_groups = 0
    if opts is None:
        self._log("Heatmap cancelled; tables will still be written.")
    else:
        try:
            fig, n_groups = self._build_cohort_heatmap_figure(names, groups, matrix, cmap_spec=opts["cmap"])
            base = build_filename("COHORT_HEATMAP", self.experiment_id, "", ext="")
            if opts.get("pdf"):
                out_pdf = os.path.join(out_dir, base + ".pdf")
                fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
                self._log(f"Wrote {out_pdf}")
            if opts.get("png"):
                out_png = os.path.join(out_dir, base + ".png")
                fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight")
                self._log(f"Wrote {out_png}")
            self._show_heatmap_window(fig, title=f"Cohort heatmap — {self.experiment_id}")
        except Exception as exc:
            self._log(f"Cohort heatmap failed: {exc}")

    try:
        written = write_tables(out_dir, self.experiment_id, names, groups, matrix)
        for w in written:
            self._log(f"Wrote {w}")
    except Exception as exc:
        self._log(f"Table export failed: {exc}")

    cond_note = (f"\n{n_groups} conditions color-coded on the heatmap." if n_groups and n_groups > 1 else "")
    messagebox.showinfo(
        "Done",
        f"Processed {len(profiles)} usable straightened sample(s).\n"
        f"Output in:\n{out_dir}{cond_note}\n\n"
        f"Analysis tables and heatmaps are in native quantitative units. "
        f"Saved TIFF storage units were divided by {SCAMP_ZDEPTH_UINT16_SCALE:g} before profile extraction."
    )
    self._log(f"FINISHED: {len(profiles)} native-unit profiles → {out_dir}")
    self._log(
        f"Analysis-unit policy: generated Z-depth TIFFs are stored as native ×{SCAMP_ZDEPTH_UINT16_SCALE:g}, "
        f"but Process all divides by {SCAMP_ZDEPTH_UINT16_SCALE:g} before writing CSV/XLSX/heatmap values."
    )


App.process_all = _scamp_process_all_native_units


if __name__ == "__main__":
    main()
