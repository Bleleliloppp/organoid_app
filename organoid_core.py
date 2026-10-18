"""
organoid_core.py
----------------
Read images (.nd2 / .tif / .png), segment the organoid, build the skeleton and compute
morphology metrics to compare "single tube" with "branched".

Each image is assumed to contain only 1 organoid (the largest region is kept).
"""
from __future__ import annotations

import io
import math
import os
import re
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import networkx as nx
from scipy import ndimage as ndi
from skimage import filters, measure, transform

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------


@dataclass
class Params:
    # Organoid segmentation
    seg_method: str = "combo"       # "combo" (texture or dark) | "tube" (ridge filter) | "texture" | "dark" | "bright"
    tube_width_um: float = 0.0      # "tube" method: typical tube width (µm), 0 = try a range of widths automatically
    texture_um: float = 5.0         # radius of the region used to measure "roughness" (µm)
    threshold_scale: float = 1.0    # >1: tighter mask, <1: looser mask
    smooth_um: float = 5.0          # mask edge smoothing (µm)
    edge_shrink: float = 2.0        # shrink edge inward (× filter radius) to compensate for the filter "bleeding" into the background
    low_ratio: float = 0.6          # fainter parts connected to clear parts are kept if > low_ratio × threshold (1 = off)
    join_gap_um: float = 30.0       # join large pieces no farther than this from the main piece (µm), 0 = off
    hole_factor: float = 1.0        # holes narrower than (factor × tube width) are filled; wider holes are real lumens/loops
    # Skeleton
    spur_factor: float = 1.0        # dead-end branches shorter than (factor × tube width) are treated as spurs
    min_branch_um: float = 0.0      # absolute minimum branch length (µm), 0 = ignore
    # Classification
    single_tube_min_fraction: float = 0.8   # "single tube" = no loops and main path ≥ this fraction of the skeleton
    # Image reading
    max_dim: int = 2048             # larger images are downscaled for analysis (0 = no downscaling)
    z_mode: str = "edf"             # "edf" (merge slices, take sharpest regions) | "best" (single sharpest slice) | "index"
    z_index: int = 0
    channel: int = -1               # -1 = auto-pick green fluorescence channel (GFP/FITC/488), ≥0 = channel number (from 0)
    position: int = 0
    time: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class LoadedImage:
    gray: np.ndarray                # 2D float32 image, 0..1
    px_um: float | None             # µm / pixel (after downscaling, if any)
    source: str
    notes: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


# Key metrics used to compare groups
KEY_METRICS = [
    "is_single_tube",
    "main_path_fraction",
    "n_junctions",
    "junctions_per_mm",
    "n_cycles",
    "solidity",
]

# Yes/no metrics: compared as proportions (Fisher / chi-square) instead of medians
BINARY_METRICS = {"is_single_tube"}

METRIC_LABELS = {
    "is_single_tube": "Single tube (1 = yes, 0 = branched)",
    "junctions_per_mm": "Junctions per mm of skeleton",
    "main_path_fraction": "Main path / total skeleton",
    "n_junctions": "Junctions",
    "n_endpoints": "Endpoints",
    "n_branches": "Branch segments",
    "n_cycles": "Closed loops",
    "tortuosity": "Main path tortuosity",
    "rms_dev_rel": "RMS deviation / length",
    "max_dev_rel": "Max deviation / length",
    "rms_dev_um": "RMS deviation (µm)",
    "main_length_um": "Main path length (µm)",
    "end_to_end_um": "End-to-end distance (µm)",
    "skeleton_length_um": "Total skeleton length (µm)",
    "longest_side_um": "Longest side branch (µm)",
    "longest_side_rel": "Longest side branch / main path",
    "width_mean_um": "Mean width along main path (µm)",
    "width_cv": "Width CV",
    "area_um2": "Area (µm²)",
    "perimeter_um": "Perimeter (µm)",
    "solidity": "Solidity",
    "aspect_ratio": "Aspect ratio",
    "circularity": "Circularity",
    "px_um": "µm / pixel",
}

# ----------------------------------------------------------------------------
# Image reading
# ----------------------------------------------------------------------------


def load_image(path: str, params: Params | None = None,
               px_um_override: float | None = None) -> LoadedImage:
    """Read an image, convert to 2D grayscale 0..1, with µm/pixel if available."""
    p = params or Params()
    ext = os.path.splitext(path)[1].lower()
    if ext == ".nd2":
        arr, dims, px, notes, info = _read_nd2(path)
    elif ext in (".tif", ".tiff"):
        arr, dims, px, notes, info = _read_tiff(path)
    else:
        import imageio.v3 as iio
        arr = np.asarray(iio.imread(path))
        dims = _guess_dims(arr.shape)
        px, notes, info = None, [], {}

    n_ch = next((arr.shape[i] for i, d in enumerate(dims) if str(d).upper() == "C"), 1)
    ch, ch_note = _pick_channel(p.channel, n_ch, info)
    if ch_note:
        notes.append(ch_note)
    gray, more, pre_scale = _reduce_to_2d(arr, dims, replace(p, channel=ch))
    notes += more
    gray = _normalize(gray)

    if px_um_override:
        if px is not None and abs(px - px_um_override) > 1e-6:
            notes.append(f"Using manually entered scale {px_um_override:.4g} µm/px "
                         f"(metadata says {px:.4g} µm/px).")
        px = float(px_um_override)

    if px is None:
        notes.append("No µm/pixel information. Length measurements are in pixels; "
                     "ratios (fraction, tortuosity, solidity...) are still valid.")

    yx = [arr.shape[i] for i, d in enumerate(dims) if str(d).upper() in ("Y", "X")]
    info["original_shape"] = tuple(yx) if len(yx) == 2 else tuple(gray.shape)
    scale = pre_scale
    if p.max_dim and max(gray.shape) > p.max_dim:
        f = p.max_dim / max(gray.shape)
        gray = transform.resize(gray, (round(gray.shape[0] * f), round(gray.shape[1] * f)),
                                anti_aliasing=True, preserve_range=True).astype(np.float32)
        scale *= f
    if scale < 1:
        if px is not None:
            px = px / scale
        notes.append(f"Image downscaled to {gray.shape[1]}×{gray.shape[0]} px for analysis "
                     f"(scale adjusted accordingly).")
    info["analysis_shape"] = tuple(gray.shape)
    return LoadedImage(gray=gray, px_um=px, source=os.path.basename(path), notes=notes, info=info)


def _read_nd2(path):
    try:
        import nd2
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("The .nd2 reader library is not installed. Run: pip install nd2") from e
    notes, info = [], {}
    px = None
    with nd2.ND2File(path) as f:
        sizes = dict(f.sizes)
        arr = np.asarray(f.asarray())
        try:
            vs = f.voxel_size()
            px_x, px_y = float(vs.x), float(vs.y)
            if px_x > 0:
                px = px_x
                if abs(px_x - px_y) > 1e-3 * px_x:
                    notes.append(f"Non-square pixels (x={px_x:.4g}, y={px_y:.4g} µm); using x.")
                if px_x == 1.0 and px_y == 1.0:
                    notes.append("Metadata says exactly 1.0 µm/px, the image may not be calibrated. "
                                 "Check it in Fiji (Image → Properties).")
        except Exception as e:  # metadata missing or corrupt
            notes.append(f"Could not read µm/pixel from metadata: {e}")
        try:
            chans = f.metadata.channels or []
            info["channels"] = [c.channel.name for c in chans]
            info["emission_nm"] = [getattr(c.channel, "emissionLambdaNm", None) for c in chans]
            mic = chans[0].microscope
            info["objective"] = getattr(mic, "objectiveName", None)
            info["objective_mag"] = getattr(mic, "objectiveMagnification", None)
        except Exception:
            pass
    dims = [str(d).upper() for d in sizes.keys()]
    if tuple(arr.shape) != tuple(sizes.values()):
        dims = _guess_dims(arr.shape)
        notes.append(f"Dimension order {sizes} does not match data {arr.shape}; guessing {dims}.")
    info["dims"] = dict(zip(dims, arr.shape))
    return arr, dims, px, notes, info


def _read_tiff(path):
    import tifffile
    notes, info = [], {}
    px = None
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        arr = np.asarray(s.asarray())
        axes = s.axes.upper()
        dims = ["S" if a in "S" else a for a in axes]
        if len(dims) != arr.ndim:
            dims = _guess_dims(arr.shape)
        # OME-TIFF
        if tf.ome_metadata:
            m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', tf.ome_metadata)
            if m:
                px = float(m.group(1))
        # ImageJ / Fiji TIFF
        if px is None:
            try:
                ij = tf.imagej_metadata or {}
                unit = str(ij.get("unit", "")).lower()
                xr = tf.pages[0].tags["XResolution"].value
                if unit in ("micron", "um", "µm", "\\u00b5m", "microns") and xr[0] > 0:
                    px = xr[1] / xr[0]
            except Exception:
                pass
    info["dims"] = dict(zip(dims, arr.shape))
    return arr, dims, px, notes, info


_GREEN_WORDS = ("gfp", "fitc", "488", "green", "cy2", "alexa 488", "af488", "mneon", "venus", "yfp")
_TRANS_WORDS = ("bf", "brightfield", "bright field", "dic", "td", "trans", "phase", "ph")


def _pick_channel(want, n_ch, info):
    """Returns (channel number, note). want < 0: auto-pick the green channel (GFP/FITC/488) by name or
    emission wavelength; if none is recognized, take the first fluorescence channel (skip brightfield/DIC)."""
    names = list(info.get("channels") or [])
    em = list(info.get("emission_nm") or [])
    label = lambda k: (f"channel {k + 1}/{n_ch}" + (f" ('{names[k]}')" if k < len(names) and names[k] else ""))
    if n_ch <= 1:
        return 0, None
    if want >= 0:
        k = int(min(want, n_ch - 1))
        return k, f"Using {label(k)} (manually selected)."
    low = [str(n).lower() for n in names[:n_ch]]
    for k, n in enumerate(low):
        if any(w in n for w in _GREEN_WORDS):
            return k, f"Auto-selected {label(k)} because its name is a green/GFP channel."
    for k, e in enumerate(em[:n_ch]):
        if e and 495 <= float(e) <= 570:
            return k, f"Auto-selected {label(k)} because its emission wavelength is {float(e):.0f} nm (green)."
    is_trans = lambda n: any(n == w or n.startswith(w + " ") or w in n.split() for w in _TRANS_WORDS)
    k = next((k for k, n in enumerate(low) if not is_trans(n)), 0)
    shown = ", ".join(f"{i}: {n}" for i, n in enumerate(names[:n_ch])) or "unnamed"
    return k, (f"No GFP channel recognized; using {label(k)}. Channels ({shown}) — "
               "if wrong, select one manually under 'Channel'.")


def _guess_dims(shape):
    if len(shape) == 2:
        return ["Y", "X"]
    if len(shape) == 3 and shape[-1] in (3, 4):
        return ["Y", "X", "S"]
    extra = ["T", "Z", "C"][-(len(shape) - 2):] if len(shape) <= 5 else ["Q"] * (len(shape) - 2)
    return list(extra) + ["Y", "X"]


def _focus_score(img2d):
    small = img2d[::2, ::2].astype(np.float32) if max(img2d.shape) > 1024 else img2d.astype(np.float32)
    return float(ndi.laplace(ndi.gaussian_filter(small, 1.0)).var())


def _reduce_to_2d(arr, dims, p: Params):
    """Select channel / position / time point / Z slice to produce a 2D image."""
    notes = []
    dims = [d.upper() for d in dims]
    index = []
    keep = []
    for i, d in enumerate(dims):
        n = arr.shape[i]
        if d in ("Y", "X", "S") or (d == "Z" and n > 1):
            index.append(slice(None))
            keep.append(d)
            continue
        want = {"C": p.channel, "P": p.position, "T": p.time, "Z": p.z_index}.get(d, 0)
        k = int(min(max(want, 0), n - 1))
        if n > 1 and d != "C":            # channel is already noted in _pick_channel
            name = {"P": "positions (multipoint)", "T": "time points"}.get(d, f"dimension {d}")
            notes.append(f"File has {n} {name}; using number {k}.")
        index.append(k)
    a = arr[tuple(index)]

    if "S" in keep:  # RGB color image
        si = keep.index("S")
        a = np.moveaxis(a, si, -1).astype(np.float32)
        keep.remove("S")
        keep.append("S")
        if a.shape[-1] >= 3:
            a = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
        else:
            a = a[..., 0]
        keep.remove("S")

    pre_scale = 1.0
    if "Z" in keep:
        zi = keep.index("Z")
        a = np.moveaxis(a, zi, 0)
        keep.remove("Z")
        nz = a.shape[0]
        if p.z_mode == "index":
            k = int(min(max(p.z_index, 0), nz - 1))
            notes.append(f"File has {nz} Z slices; using slice {k} (manually selected).")
            a = a[k]
        elif p.z_mode == "best":
            scores = [_focus_score(a[z]) for z in range(nz)]
            k = int(np.argmax(scores))
            notes.append(f"File has {nz} Z slices; using the single sharpest slice, number {k}.")
            a = a[k]
        else:  # "edf": merge slices, each region taken from its sharpest slice
            if p.max_dim and max(a.shape[1:]) > p.max_dim:
                pre_scale = p.max_dim / max(a.shape[1:])
                shp = (round(a.shape[1] * pre_scale), round(a.shape[2] * pre_scale))
                a = np.stack([transform.resize(a[z].astype(np.float32), shp, anti_aliasing=True,
                                               preserve_range=True) for z in range(nz)])
            a, used = extended_depth_of_field(a)
            notes.append(f"File has {nz} Z slices; merged them, each region taken from its sharpest slice "
                         f"(used {len(used)} slices: {', '.join(map(str, used))}).")

    # Ensure (Y, X) order
    if keep == ["X", "Y"]:
        a = a.T
    return np.asarray(a, dtype=np.float32), notes, pre_scale


def extended_depth_of_field(stack, sigma=4.0):
    """Merge a Z-stack into one image sharp everywhere: each pixel taken from the slice with the highest local sharpness.
    Returns (2D image, list of slices actually used)."""
    st = np.asarray(stack, dtype=np.float32)
    nz = st.shape[0]
    # bring slices to the same brightness level so seams do not show brightness jumps
    med = np.median(st.reshape(nz, -1), axis=1)
    st = st - med[:, None, None] + med.mean()
    focus = np.empty_like(st)
    for k in range(nz):
        lap = ndi.laplace(ndi.gaussian_filter(st[k], 1.0))
        focus[k] = ndi.gaussian_filter(lap * lap, sigma)
    idx = np.argmax(focus, axis=0)
    idx = ndi.median_filter(idx, size=9)
    out = np.take_along_axis(st, idx[None], axis=0)[0]
    counts = np.bincount(idx.ravel(), minlength=nz)
    used = [int(k) for k in np.nonzero(counts > 0.02 * idx.size)[0]]
    return out, used


def _normalize(a):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------
# Organoid segmentation
# ----------------------------------------------------------------------------


def _dilate(m, r):
    return ndi.distance_transform_edt(~m) <= r


def _erode(m, r):
    return ndi.distance_transform_edt(m) > r


def _largest_component(m):
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def _components(m):
    lab, n = ndi.label(m)
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1)) if n else np.zeros(0)
    return lab, n, np.asarray(sizes, dtype=float)


def tube_width(m):
    """Typical tube width (px) = 2 × median distance to the edge along the skeleton."""
    from skimage.morphology import skeletonize
    sk = skeletonize(m)
    if not sk.any():
        return 2.0
    return float(2 * np.median(ndi.distance_transform_edt(m)[sk]))


def fill_small_holes(m, factor=1.0, width=None):
    """Fill holes whose own width (largest inscribed circle) is below factor × tube width.
    Wider holes are real lumens between branches of a network and stay open (loops in the skeleton)."""
    m = m.astype(bool)
    holes = ndi.binary_fill_holes(m) & ~m
    if not holes.any():
        return m
    lab, n, _ = _components(holes)
    width = tube_width(m) if width is None else width
    hole_w = 2 * np.asarray(ndi.maximum(ndi.distance_transform_edt(holes), lab, np.arange(1, n + 1)))
    small = np.nonzero(hole_w < factor * width)[0] + 1
    return m | np.isin(lab, small)


def clean_mask(mask, r_px, keep_all=False):
    """Smooth the mask, fill holes smaller than the smoothing scale (larger holes are judged later,
    once the edge is in its final place). keep_all=False: keep only the largest region."""
    m = mask.astype(bool)
    w = 2 * max(r_px, 1.0)
    if r_px >= 1:
        m = _erode(_dilate(m, r_px), r_px)       # closing
        m = fill_small_holes(m, 1.0, w)
        m = _dilate(_erode(m, r_px), r_px)       # opening
    if not keep_all:
        m = _largest_component(m)
    if r_px >= 1:
        m = ndi.gaussian_filter(m.astype(np.float32), r_px / 2) > 0.5
    m = fill_small_holes(m, 1.0, w)
    return m if keep_all else _largest_component(m)


def _tube_radius(m):
    """Estimated tube radius ≈ area / perimeter (valid for long tubes)."""
    per = measure.perimeter(m)
    return float(m.sum() / per) if per > 0 else 1.0


def _thick_segment(shape, p0, p1, r):
    from skimage.draw import line as draw_line
    seed = np.zeros(shape, bool)
    rr, cc = draw_line(int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]))
    seed[rr, cc] = True
    return _dilate(seed, max(1.0, r))


def _join_pieces(lab, sizes, main_i, gap_px, min_frac=0.05):
    """Join large pieces (≥ min_frac × main piece) lying no farther than gap_px from the kept part,
    using a connector thinner than the tube width. Returns (mask, list of joined labels)."""
    keep = lab == main_i + 1
    cand = [i for i in range(len(sizes)) if i != main_i and sizes[i] >= min_frac * sizes[main_i]]
    joined = []
    while cand:
        d, inds = ndi.distance_transform_edt(~keep, return_indices=True)
        best = None
        for i in cand:
            piece = lab == i + 1
            dd = np.where(piece, d, np.inf)
            j = int(np.argmin(dd))
            if dd.flat[j] <= gap_px and (best is None or dd.flat[j] < best[1]):
                best = (i, dd.flat[j], j)
        if best is None:
            break
        i, _, j = best
        piece = lab == i + 1
        py, px_ = np.unravel_index(j, d.shape)
        q = (inds[0, py, px_], inds[1, py, px_])
        r = 0.8 * min(_tube_radius(piece), _tube_radius(keep))
        keep = keep | piece | _thick_segment(keep.shape, (py, px_), q, r)
        cand.remove(i)
        joined.append(i + 1)
    return keep, joined


def _shrink_keep_connected(m, r):
    """Shrink the edge by r pixels without letting the mask break at narrow necks."""
    if r <= 0 or not m.any():
        return m
    sh = _erode(m, r)
    lab, n, sizes = _components(sh)
    if n == 0 or sizes.max() < 0.3 * m.sum():
        return m                                   # skip if it shrinks too much
    if (sizes >= 0.05 * sizes.max()).sum() > 1:     # broken: keep the medial axis to connect the parts
        from skimage.morphology import skeletonize
        sh = sh | (_dilate(skeletonize(m), 1.5) & m)
    return _largest_component(sh)


def _padding_mask(gray, min_frac=0.002):
    """Constant black (minimum-value) regions touching the image edge, usually from stitching tiles."""
    m = gray <= gray.min()
    lab, n, sizes = _components(m)
    if n == 0:
        return np.zeros_like(m)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    keep = [l for l in edge if l > 0 and sizes[l - 1] >= min_frac * m.size]
    return np.isin(lab, keep)


def _polarity(g, tex, valid):
    """+1 if the textured region is darker than the background (brightfield), -1 if brighter (fluorescence)."""
    t = filters.threshold_otsu(tex[valid])
    obj = valid & (tex > t)
    bg = valid & ~obj
    if obj.sum() < 10 or bg.sum() < 10:
        return 1
    return 1 if np.median(g[obj]) < np.median(g[bg]) else -1


TUBE_HIGH, TUBE_LOW = 2.5, 1.45     # "tube" method: threshold multipliers (high, faint parts)


def segment(gray, px_um, p: Params):
    """Returns (mask, feature, info). The mask keeps only the main organoid (plus joined pieces)."""
    px = px_um or 1.0
    s_tex = max(1.0, p.texture_um / px)
    pad = _padding_mask(gray)
    valid = ~pad
    g = gray.astype(np.float32).copy()
    if pad.any():
        g[pad] = np.median(g[valid])   # fill black stitching areas so they do not create fake edges / "darkness"
    g = ndi.gaussian_filter(g, 1.0)

    def norm(f):
        lo, hi = np.percentile(f[valid], [1, 99.9])
        out = np.clip((f - lo) / max(hi - lo, 1e-9), 0, 1)
        out[pad] = 0
        return out

    def texture():
        m1 = ndi.gaussian_filter(g, s_tex)
        m2 = ndi.gaussian_filter(g * g, s_tex)
        return np.sqrt(np.clip(m2 - m1 * m1, 0, None))

    def contrast(sigma, sign):
        bg = ndi.gaussian_filter(g, max(g.shape) / 6)
        return sign * (bg - ndi.gaussian_filter(g, sigma))

    if p.seg_method == "texture":
        sigma_used = s_tex
        feat = norm(texture())
    elif p.seg_method in ("dark", "bright"):
        sigma_used = max(1.0, s_tex / 2)
        feat = norm(contrast(sigma_used, 1 if p.seg_method == "dark" else -1))
    elif p.seg_method == "combo":
        # texture OR differs from background: also catches blurred parts (no texture but still differ from background).
        # Auto-detect whether the organoid is darker (brightfield) or brighter (fluorescence) than the background.
        sigma_used = s_tex
        tex = norm(texture())
        sign = _polarity(g, tex, valid)
        feat = np.maximum(tex, norm(contrast(s_tex, sign)))
    elif p.seg_method == "tube":
        # ridge (tubeness) filter: strong on elongated tubes, weak on round blobs and grainy cell clumps
        sign = _polarity(g, norm(texture()), valid)
        if p.tube_width_um > 0:
            r = p.tube_width_um / px / 2
        else:
            # estimate the tube width with a quick first pass over a wide range of scales
            first = norm(filters.sato(g, sigmas=np.geomspace(2.0, max(3.0, min(g.shape) / 60), 5),
                                      black_ridges=sign > 0))
            m0 = _largest_component(first > filters.threshold_otsu(first[valid]))
            r = tube_width(m0) / 2
        sigmas = np.geomspace(max(1.0, r / 3), max(1.5, r), 4)
        sigma_used = float(sigmas[0])
        feat = norm(filters.sato(g, sigmas=sigmas, black_ridges=sign > 0))
    else:
        raise ValueError(f"Invalid segmentation method: {p.seg_method}")
    t = filters.threshold_otsu(feat[valid]) * p.threshold_scale
    low = max(p.low_ratio, 0.05) * t
    if p.seg_method == "tube":
        # ridge response is skewed: only strong ridges start the mask, weaker ridges connected to them are
        # kept (calibrated so the default settings trace thin connecting tubes but not ridges in cell clumps)
        t, low = TUBE_HIGH * t, TUBE_LOW * low
    if p.low_ratio < 1:
        # fainter regions connected to clear regions are still kept (hysteresis)
        raw = filters.apply_hysteresis_threshold(feat, low, t)
    else:
        raw = feat > t
    allm = clean_mask(raw, p.smooth_um / px, keep_all=True)

    info = {"n_pieces": 0, "joined": 0, "dropped": np.zeros_like(allm), "dropped_frac": 0.0}
    lab, n, sizes = _components(allm)
    if n == 0:
        return allm, feat, info
    main_i = int(np.argmax(sizes))
    keep = lab == main_i + 1
    kept_labels = [main_i + 1]
    if p.join_gap_um > 0 and n > 1:
        keep, joined = _join_pieces(lab, sizes, main_i, p.join_gap_um / px)
        kept_labels += joined
        info["joined"] = len(joined)
    if p.edge_shrink > 0:
        keep = _shrink_keep_connected(keep, p.edge_shrink * sigma_used)
    keep = fill_small_holes(keep, p.hole_factor)   # after shrinking, when holes have their real size

    # Dropped pieces (only significant ones, ≥ 5% of the main piece, to avoid reporting debris)
    sig = [i + 1 for i in range(n) if sizes[i] >= 0.05 * sizes[main_i]]
    dropped_labels = [l for l in sig if l not in kept_labels]
    info["n_pieces"] = len(sig)
    if dropped_labels:
        info["dropped"] = np.isin(lab, dropped_labels)
        info["dropped_frac"] = float(sum(sizes[l - 1] for l in dropped_labels) /
                                     sum(sizes[l - 1] for l in kept_labels))
    return keep, feat, info


def load_mask(path, shape, hole_factor=1.0):
    """Read an existing mask (from Fiji / ilastik / Cellpose...) and resize it to match."""
    import imageio.v3 as iio
    m = np.asarray(iio.imread(path))
    if m.ndim == 3:
        m = m[..., 0] if m.shape[-1] in (3, 4) else m[0]
    m = m > 0
    if m.shape != tuple(shape):
        m = transform.resize(m.astype(np.float32), shape, order=0, preserve_range=True) > 0.5
    m = fill_small_holes(m, hole_factor)
    return _largest_component(m)


# ----------------------------------------------------------------------------
# Skeleton → graph
# ----------------------------------------------------------------------------


def skeleton_graph(skel):
    """Pixel graph of the skeleton (8-neighbour, redundant diagonal edges dropped to avoid fake triangles)."""
    ys, xs = np.nonzero(skel)
    n = len(ys)
    coords = np.stack([ys, xs], axis=1)
    idx = -np.ones(skel.shape, dtype=np.int64)
    idx[ys, xs] = np.arange(n)
    pad = np.pad(skel, 1)
    pidx = np.pad(idx, 1, constant_values=-1)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ny, nx_ = ys + 1 + dy, xs + 1 + dx
        ok = pad[ny, nx_]
        if dy and dx:  # diagonal edge: drop if the 2 pixels are already connected through a shared 4-neighbour
            ok &= ~(pad[ys + 1 + dy, xs + 1] | pad[ys + 1, xs + 1 + dx])
            w = math.sqrt(2)
        else:
            w = 1.0
        a = np.arange(n)[ok]
        b = pidx[ny[ok], nx_[ok]]
        G.add_weighted_edges_from(zip(a.tolist(), b.tolist(), [w] * len(a)))
    return G, coords


def prune_spurs(G, thresh_px):
    """Prune dead-end branches (endpoint → junction) shorter than thresh_px, repeating until stable."""
    G = G.copy()
    while True:
        spurs = {}  # junction -> list of (length, path)
        for e in [v for v, d in G.degree() if d == 1]:
            path, length, prev, cur, junction = [e], 0.0, None, e, None
            while True:
                nbrs = [v for v in G.neighbors(cur) if v != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                length += G[cur][nxt]["weight"]
                if length >= thresh_px:
                    break
                d = G.degree(nxt)
                if d >= 3:
                    junction = nxt
                    break
                if d == 1:
                    break  # the whole segment is a single line, do not prune
                path.append(nxt)
                prev, cur = cur, nxt
            if junction is not None:
                spurs.setdefault(junction, []).append((length, path))
        remove = set()
        for j, lst in spurs.items():
            lst.sort(key=lambda t: t[0])
            if G.degree(j) - len(lst) <= 0:   # all branches around this point are short → keep the longest
                lst = lst[:-1]
            for _, path in lst:
                remove.update(path)
        if not remove:
            break
        G.remove_nodes_from(remove)
    if G.number_of_nodes() and nx.number_connected_components(G) > 1:
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


def _resample(pts, step=1.0):
    seg = np.sqrt((np.diff(pts, axis=0) ** 2).sum(1))
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] <= 0:
        return pts, s
    u = np.arange(0, s[-1] + 1e-9, step)
    return np.stack([np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])], axis=1), u


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------


def analyze(gray, px_um, p: Params | None = None, mask=None):
    """Run the full pipeline on one 2D image. Returns a dict with 'metrics' and layers for drawing."""
    p = p or Params()
    px = px_um or 1.0
    unit_ok = px_um is not None
    warnings = []
    dropped = None
    if mask is None:
        mask, feat, seg = segment(gray, px_um, p)
        if seg["dropped_frac"] > 0:
            dropped = seg["dropped"]
            warnings.append(
                f"The mask is broken into {seg['n_pieces']} large pieces; the app is dropping about "
                f"{seg['dropped_frac'] * 100:.0f}% of the area relative to the kept part (dashed red outline on the image). "
                "If the dropped part is a section of the organoid: decrease 'Threshold for faint parts', decrease 'Threshold factor', "
                "or increase 'Join pieces up to this distance apart'.")
        if seg["joined"]:
            warnings.append(f"Joined {seg['joined']} pieces to the main organoid. Check on the image that the joins are correct.")
        if mask.any() and (~mask).any():
            fo = feat[~mask]
            contrast = (np.median(feat[mask]) - np.median(fo)) / (fo.std() + 1e-9)
            if contrast < 1.5:
                warnings.append("Low contrast between organoid and background, the mask may be wrong. "
                                "Review the overlay or change the segmentation method.")
    mask = mask.astype(bool)
    if mask.sum() < 50:
        raise ValueError("Could not segment the organoid (mask too small). Try decreasing the threshold factor "
                         "or changing the segmentation method.")
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]]).mean()
    if border > 0.02:
        warnings.append("The organoid touches the image edge and may be partly cut off, so length and "
                        "branch count may be underestimated.")

    from skimage.morphology import skeletonize
    dist = ndi.distance_transform_edt(mask)
    skel = skeletonize(mask)
    G0, coords = skeleton_graph(skel)
    radii = dist[coords[:, 0], coords[:, 1]]
    median_diam_px = float(2 * np.median(radii[radii > 0])) if np.any(radii > 0) else 0.0
    thresh_px = max(p.spur_factor * median_diam_px, p.min_branch_um / px)
    G = prune_spurs(G0, thresh_px)

    deg = dict(G.degree())
    ends = [v for v, d in deg.items() if d == 1]
    junc_px = [v for v, d in deg.items() if d >= 3]
    junc_clusters = list(nx.connected_components(G.subgraph(junc_px)))
    V, E = G.number_of_nodes(), G.number_of_edges()
    C = nx.number_connected_components(G) if V else 0
    n_cycles = E - V + C
    total_len_px = G.size(weight="weight")
    vb = len(ends) + len(junc_clusters)
    n_branches = (vb - C + n_cycles) if vb else n_cycles

    # Main path = longest path between 2 endpoints
    main_path = []
    main_len_px = float("nan")
    if len(ends) >= 2:
        best = (-1.0, None, None)
        for e in ends:
            lengths = nx.single_source_dijkstra_path_length(G, e, weight="weight")
            for f in ends:
                if f != e and lengths.get(f, -1) > best[0]:
                    best = (lengths[f], e, f)
        main_len_px, a, b = best
        main_path = nx.shortest_path(G, a, b, weight="weight")
    elif V:
        main_len_px = total_len_px

    m = {}
    m["px_um"] = px_um if unit_ok else float("nan")
    m["n_junctions"] = len(junc_clusters)
    m["n_endpoints"] = len(ends)
    m["n_branches"] = int(n_branches)
    m["n_cycles"] = int(n_cycles)
    m["skeleton_length_um"] = total_len_px * px
    m["main_length_um"] = main_len_px * px
    m["main_path_fraction"] = (main_len_px / total_len_px) if total_len_px > 0 else float("nan")
    m["junctions_per_mm"] = len(junc_clusters) / (total_len_px * px / 1000) if total_len_px > 0 else float("nan")
    # Single tube = no closed loops and the main path covers most of the skeleton
    single = (n_cycles == 0 and len(main_path) > 0
              and m["main_path_fraction"] >= p.single_tube_min_fraction)
    m["is_single_tube"] = int(single)

    main_pts = coords[main_path].astype(np.float64) if main_path else np.zeros((0, 2))
    if len(main_pts) >= 3:
        pts, s = _resample(main_pts, 1.0)
        chord = float(np.linalg.norm(main_pts[-1] - main_pts[0]))
        m["end_to_end_um"] = chord * px
        m["tortuosity"] = (main_len_px / chord) if chord > 0 else float("nan")
        c = pts - pts.mean(0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        perp = c @ vt[1]
        rms = float(np.sqrt(np.mean(perp ** 2)))
        m["rms_dev_um"] = rms * px
        m["rms_dev_rel"] = rms / main_len_px
        m["max_dev_rel"] = float(np.abs(perp).max()) / main_len_px
        r_main = dist[main_pts[:, 0].astype(int), main_pts[:, 1].astype(int)]
        seg = np.sqrt((np.diff(main_pts, axis=0) ** 2).sum(1))
        s_main = np.concatenate([[0], np.cumsum(seg)])
        r_med = float(np.median(r_main))
        inner = (s_main > r_med) & (s_main < s_main[-1] - r_med)
        rr = r_main[inner] if inner.sum() >= 3 else r_main
        w = 2 * rr * px
        m["width_mean_um"] = float(w.mean())
        m["width_cv"] = float(w.std() / w.mean()) if w.mean() > 0 else float("nan")
    else:
        for k in ("end_to_end_um", "tortuosity", "rms_dev_um", "rms_dev_rel", "max_dev_rel",
                  "width_mean_um", "width_cv"):
            m[k] = float("nan")
        warnings.append("The organoid is nearly round (no long axis), so the "
                        "main path metrics are left empty.")

    # Longest side branch (each "subtree" attached to the main path)
    side_nodes = set(G.nodes) - set(main_path)
    longest_side = 0.0
    if side_nodes:
        main_set = set(main_path)
        for comp in nx.connected_components(G.subgraph(side_nodes)):
            L = G.subgraph(comp).size(weight="weight")
            L += max((G[u][v]["weight"] for u in comp for v in G.neighbors(u) if v in main_set),
                     default=0.0)
            longest_side = max(longest_side, L)
    m["longest_side_um"] = longest_side * px
    m["longest_side_rel"] = (longest_side / main_len_px) if main_len_px and main_len_px > 0 else float("nan")

    rp = measure.regionprops(mask.astype(np.uint8))[0]
    area, per = float(rp.area), float(rp.perimeter)
    m["area_um2"] = area * px * px
    m["perimeter_um"] = per * px
    m["solidity"] = float(rp.solidity)
    m["aspect_ratio"] = float(rp.axis_major_length / rp.axis_minor_length) if rp.axis_minor_length > 0 else float("nan")
    m["circularity"] = float(4 * math.pi * area / per ** 2) if per > 0 else float("nan")

    side_idx = np.array(sorted(side_nodes), dtype=np.int64)
    return {
        "metrics": m,
        "mask": mask,
        "main_path_yx": main_pts,
        "side_yx": coords[side_idx] if len(side_idx) else np.zeros((0, 2), int),
        "junctions_yx": np.array([coords[list(c)].mean(0) for c in junc_clusters]) if junc_clusters else np.zeros((0, 2)),
        "endpoints_yx": coords[ends] if ends else np.zeros((0, 2), int),
        "spur_threshold_um": thresh_px * px,
        "units_are_um": unit_ok,
        "shape_class": "single tube" if single else "branched / network",
        "warnings": warnings,
        "dropped": dropped,
    }


def analyze_file(path, p: Params | None = None, px_um_override=None, mask_path=None):
    p = p or Params()
    img = load_image(path, p, px_um_override)
    mask = load_mask(mask_path, img.gray.shape, p.hole_factor) if mask_path else None
    res = analyze(img.gray, img.px_um, p, mask=mask)
    return img, res


# ----------------------------------------------------------------------------
# Overlay drawing
# ----------------------------------------------------------------------------

MAIN_COLOR = "#19D3E6"   # cyan: main path
SIDE_COLOR = "#FF8A1F"   # orange: side branches
OUTLINE_COLOR = "#FFE14D"
DROPPED_COLOR = "#FF3B5C"  # red: dropped pieces


def _nice_bar(width_um):
    target = width_um / 5
    for v in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000, 10000):
        if v >= target:
            return v
    return 10000


def render_overlay(gray, res, px_um, title=None, dpi=110):
    """Draw image + mask outline + skeleton. Single tube: main path (cyan) + side branches (orange).
    Branched / network: the whole skeleton in orange, since one "main path" would be misleading.
    Returns PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    h, w = gray.shape
    fig_w = 7.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * h / w), dpi=dpi)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
    ax.contour(res["mask"].astype(float), levels=[0.5], colors=[OUTLINE_COLOR], linewidths=0.8, alpha=0.8)
    dropped = res.get("dropped")
    has_dropped = dropped is not None and dropped.any()
    if has_dropped:
        ax.contour(dropped.astype(float), levels=[0.5], colors=[DROPPED_COLOR], linewidths=1.2,
                   linestyles="dashed")
    lw = max(1.6, 2.2 * 7.0 * dpi / max(w, 1) / 1.5)
    single = res.get("shape_class", "single tube") == "single tube"
    side = res["side_yx"]
    if len(side):
        ov = np.zeros((h, w), bool)
        ov[side[:, 0], side[:, 1]] = True
        r = max(1, int(round(max(h, w) / 700)))
        ov = _dilate(ov, r)
        rgba = np.zeros((h, w, 4), np.float32)
        rgba[ov] = matplotlib.colors.to_rgba(SIDE_COLOR)
        ax.imshow(rgba, interpolation="nearest")
    mp = res["main_path_yx"]
    if len(mp):
        ax.plot(mp[:, 1], mp[:, 0], color=MAIN_COLOR if single else SIDE_COLOR, lw=lw if single else lw * 0.6,
                solid_capstyle="round")
    j = res["junctions_yx"]
    if len(j):
        ax.scatter(j[:, 1], j[:, 0], s=36, c=SIDE_COLOR, edgecolors="white", linewidths=0.8, zorder=5)
    e = res["endpoints_yx"]
    if len(e):
        ax.scatter(e[:, 1], e[:, 0], s=14, c="white", edgecolors="black", linewidths=0.5, zorder=5)

    if px_um:
        bar_um = _nice_bar(w * px_um)
        bar_px = bar_um / px_um
        x0, y0 = w * 0.95 - bar_px, h * 0.94
        ax.plot([x0, x0 + bar_px], [y0, y0], color="white", lw=3, solid_capstyle="butt")
        ax.text(x0 + bar_px / 2, y0 - h * 0.015, f"{bar_um:g} µm", color="white",
                ha="center", va="bottom", fontsize=9)

    if single:
        handles = [Line2D([], [], color=MAIN_COLOR, lw=2.5, label="Main path"),
                   Line2D([], [], color=SIDE_COLOR, lw=2.5, label="Side branch")]
    else:
        handles = [Line2D([], [], color=SIDE_COLOR, lw=2.5, label="Skeleton (branched)")]
    handles += [Line2D([], [], marker="o", ls="", markerfacecolor=SIDE_COLOR, markeredgecolor="white",
                      markersize=7, label="Junction"),
               Line2D([], [], color=OUTLINE_COLOR, lw=1, label="Organoid outline")]
    if has_dropped:
        handles.append(Line2D([], [], color=DROPPED_COLOR, lw=1.2, ls="--", label="Dropped part"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.75)
    m = res.get("metrics", {})
    verdict = (f"{res['shape_class'].upper()} · main path {m.get('main_path_fraction', float('nan')):.2f} of skeleton · "
               f"{m.get('n_junctions', 0)} junctions · {m.get('n_cycles', 0)} loops") if "shape_class" in res else ""
    ax.set_title("\n".join(t for t in (title, verdict) if t), fontsize=10)
    ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def mask_png(mask):
    import imageio.v3 as iio
    buf = io.BytesIO()
    iio.imwrite(buf, (mask.astype(np.uint8) * 255), extension=".png")
    return buf.getvalue()
