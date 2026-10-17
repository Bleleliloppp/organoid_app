"""
app.py — local user interface.   Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

import batch as bt
import organoid_core as oc

st.set_page_config(page_title="Organoid analysis", layout="wide")

METHOD_LABELS = {
    "combo": "Texture or differs from background, auto bright/dark (default)",
    "texture": "Texture only",
    "dark": "Darkness only (organoid darker than background)",
    "bright": "Organoid brighter than background",
}

Z_LABELS = {
    "edf": "Merge slices: each region taken from its sharpest slice (default)",
    "best": "Use only the single sharpest slice",
    "index": "Pick one slice by number",
}

METRIC_HELP = {
    "main_path_fraction": "Single tube ≈ 1. The more branches, the lower. Key metric for 'does it form a single tube'.",
    "n_junctions": "Single tube = 0.",
    "n_endpoints": "Single tube = 2.",
    "n_branches": "Number of segments between endpoints and junctions. Single tube = 1.",
    "n_cycles": "Usually = 0. Non-zero means the skeleton has closed loops (branches touching each other).",
    "tortuosity": "Main path length / straight distance between its two ends. Straight = 1.",
    "rms_dev_rel": "RMS deviation from the best-fit straight line, divided by length. Straight = 0.",
    "max_dev_rel": "Maximum deviation from the best-fit straight line, divided by length.",
    "rms_dev_um": "RMS deviation in µm.",
    "main_length_um": "Length of the longest path through the organoid.",
    "end_to_end_um": "Straight distance between the two ends of the main path.",
    "skeleton_length_um": "Total length of all branches.",
    "longest_side_um": "Length of the longest side branch (including its sub-branches).",
    "longest_side_rel": "Longest side branch relative to the main path.",
    "width_mean_um": "Mean width along the main path (ends excluded).",
    "width_cv": "Width variability. Uniform tube ≈ 0.",
    "area_um2": "Mask area.",
    "perimeter_um": "Mask perimeter.",
    "solidity": "Area / convex hull area. Note: drops both with branching and with a curved tube.",
    "aspect_ratio": "Major axis / minor axis of the fitted ellipse.",
    "circularity": "4π·area/perimeter². Circle = 1.",
    "px_um": "Scale used for measurement.",
}

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "organoid_app_uploads")


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def save_upload(up) -> str:
    """Save a dropped file into a local temp folder (rewritten only when the file changes)."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    data = up.getvalue()
    h = hashlib.sha1(data[:1_000_000] + str(len(data)).encode()).hexdigest()[:12]
    path = os.path.join(UPLOAD_DIR, f"{h}_{os.path.basename(up.name)}")
    if not os.path.exists(path):
        with open(path, "wb") as fh:
            fh.write(data)
    return path


@st.cache_data(show_spinner=False, max_entries=8)
def cached_load(path, read_params: dict, px_override):
    p = oc.Params.from_dict(read_params)
    return oc.load_image(path, p, px_override)


def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    av = abs(v)
    if av >= 1000:
        return f"{v:,.0f}"
    if av >= 10:
        return f"{v:.1f}"
    return f"{v:.3f}"


def metrics_table(m: dict, units_are_um: bool) -> pd.DataFrame:
    rows = []
    for k, v in m.items():
        label = oc.METRIC_LABELS.get(k, k)
        if not units_are_um:
            label = label.replace("(µm²)", "(px²)").replace("(µm)", "(px)")
        rows.append({"Metric": label, "CSV column": k, "Value": fmt(v), "Meaning": METRIC_HELP.get(k, "")})
    return pd.DataFrame(rows).set_index("Metric")


# ----------------------------------------------------------------------------
# Sidebar: parameters
# ----------------------------------------------------------------------------

def sidebar_params():
    sb = st.sidebar
    sb.header("Parameters")
    defaults = oc.Params()
    px_default = None
    pj = sb.file_uploader("Load saved parameters (.json)", type=["json"])
    if pj is not None:
        try:
            d = json.loads(pj.getvalue().decode("utf-8"))
            defaults = oc.Params.from_dict(d)
            px_default = d.get("px_um_override")
        except Exception as e:
            sb.error(f"Could not read parameter file: {e}")

    sb.subheader("Organoid segmentation")
    methods = list(METHOD_LABELS)
    seg_method = sb.selectbox("Segmentation method", methods,
                              index=methods.index(defaults.seg_method) if defaults.seg_method in methods else 0,
                              format_func=METHOD_LABELS.get)
    texture_um = sb.slider("Filter radius (µm)", 1.0, 30.0, float(defaults.texture_um), 0.5,
                           help="Size of the region used to detect cell texture. Increase if the mask is patchy, decrease if it bleeds into the background.")
    threshold_scale = sb.slider("Threshold factor", 0.3, 3.0, float(defaults.threshold_scale), 0.05,
                                help="Increase if the mask spills into the background, decrease if it falls short of the organoid.")
    smooth_um = sb.slider("Edge smoothing (µm)", 0.0, 30.0, float(defaults.smooth_um), 0.5,
                          help="Too large will erase branches thinner than about 2× this value.")
    edge_shrink = sb.slider("Shrink edge inward", 0.0, 4.0, float(defaults.edge_shrink), 0.25,
                            help="Compensates for the filter spreading outward. Adjust until the yellow outline hugs the organoid edge.")
    low_ratio = sb.slider("Threshold for faint parts", 0.2, 1.0, float(defaults.low_ratio), 0.05,
                          help="Fainter or more transparent parts of the organoid, as long as they connect to the clear part, "
                               "are kept if they reach this fraction of the main threshold. Decrease if a section is lost; "
                               "increase if the mask bleeds into the background. 1 = off.")
    join_gap_um = sb.slider("Join pieces up to this distance apart (µm)", 0.0, 300.0, float(defaults.join_gap_um), 5.0,
                            help="If the mask is broken (dropped parts have a dashed red outline), large pieces closer than "
                                 "this distance are joined to the main organoid. 0 = off.")

    sb.subheader("Skeleton")
    spur_factor = sb.slider("Branch threshold (× tube width)", 0.0, 3.0, float(defaults.spur_factor), 0.1,
                            help="Dead-end branches shorter than (this value × mean tube width) are treated as spurs and removed. "
                                 "Keep fixed for all images when comparing.")
    min_branch_um = sb.number_input("Minimum branch length (µm)", 0.0, 5000.0, float(defaults.min_branch_um), 5.0,
                                    help="Absolute threshold, optional extra. 0 = ignore.")

    with sb.expander("Image reading and scale"):
        dims = [1024, 2048, 4096, 0]
        max_dim = st.selectbox("Downscale for analysis to", dims,
                               index=dims.index(defaults.max_dim) if defaults.max_dim in dims else 1,
                               format_func=lambda v: "No downscaling" if v == 0 else f"{v} px",
                               help="Very large stitched images run slowly; the scale is adjusted automatically when downscaling.")
        zopts = list(Z_LABELS)
        z_mode = st.radio("Files with multiple Z slices", zopts,
                          index=zopts.index(defaults.z_mode) if defaults.z_mode in zopts else 0,
                          format_func=Z_LABELS.get,
                          help="3D organoids rarely fit within a single slice. 'Merge slices' combines the sharpest part "
                               "of each region into one image, so sections at other depths are not lost.")
        z_index = st.number_input("Z slice number (when picking by number)", 0, 999, int(defaults.z_index))
        channel = st.number_input("Channel (-1 = auto-pick GFP/green channel)", -1, 99, int(defaults.channel),
                                  help="Counted from 0: channel 1/3 = 0, channel 2/3 = 1, channel 3/3 = 2. "
                                       "-1: auto-pick the green channel (GFP/FITC/488) from the .nd2 file metadata.")
        position = st.number_input("Position (multipoint)", 0, 9999, int(defaults.position))
        time = st.number_input("Time point", 0, 9999, int(defaults.time))
        manual = st.checkbox("Enter scale manually", value=px_default is not None,
                             help="Use only when metadata is wrong or missing. Original .nd2 files usually include the scale.")
        px_override = None
        if manual:
            px_override = st.number_input("µm / pixel", 0.001, 100.0, float(px_default or 1.0),
                                          0.01, format="%.4f")

    p = oc.Params(seg_method=seg_method, texture_um=texture_um, threshold_scale=threshold_scale,
                  smooth_um=smooth_um, edge_shrink=edge_shrink, low_ratio=low_ratio,
                  join_gap_um=join_gap_um, spur_factor=spur_factor,
                  min_branch_um=min_branch_um, max_dim=int(max_dim), z_mode=z_mode,
                  z_index=int(z_index), channel=int(channel), position=int(position), time=int(time))
    sb.download_button("Save parameters (.json)",
                       json.dumps({**p.to_dict(), "px_um_override": px_override}, ensure_ascii=False, indent=2),
                       file_name="parameters.json", mime="application/json",
                       help="Use exactly this set for every image so comparisons are fair.")
    return p, px_override


def read_params(p: oc.Params) -> dict:
    d = p.to_dict()
    return {k: d[k] for k in ("max_dim", "z_mode", "z_index", "channel", "position", "time")}


# ----------------------------------------------------------------------------
# Tab 1: single image
# ----------------------------------------------------------------------------

def tab_single(p: oc.Params, px_override):
    st.write("Drop in one image to see the result, and adjust the parameters in the sidebar until the yellow outline "
             "hugs the organoid. Then save the parameters and run the whole folder in the next tab.")
    c1, c2 = st.columns(2)
    # No file types declared for the uploader: Safari on Mac greys out (disallows) unusual extensions like .nd2.
    # The extension is checked below.
    up = c1.file_uploader("Image (.nd2, .tif, .png, .jpg)",
                          help="Drag and drop a file here, or click Browse files to choose one.")
    mask_up = c2.file_uploader("Existing mask (optional, .png/.tif)",
                               help="If the automatic mask is wrong, fix it in Fiji/ilastik and drop it here.")
    typed = st.text_input(
        "Or paste the path to an image file (recommended for large .nd2 files, since nothing has to be uploaded)",
        placeholder="/Users/name/organoid/a.nd2",
        help="Mac: select the file in Finder, hold Option, right-click → \"Copy … as Pathname\", "
             "then paste here. Windows: hold Shift, right-click the file → Copy as path.")
    typed = os.path.expanduser(typed.strip().strip('"').strip("'"))

    if typed:
        if not os.path.isfile(typed):
            st.error("No file found at this path. Check the pasted path.")
            return
        path, name = typed, os.path.basename(typed)
    elif up is not None:
        path, name = None, up.name
    else:
        return
    if not name.lower().endswith(bt.IMAGE_EXTS):
        st.error(f"The app cannot read \"{name}\". Please use a .nd2, .tif, .png or .jpg file.")
        return
    if mask_up is not None and not mask_up.name.lower().endswith((".png", ".tif", ".tiff")):
        st.error("The mask must be a .png or .tif file.")
        return

    try:
        if path is None:
            path = save_upload(up)
        with st.spinner("Reading image..."):
            img = cached_load(path, read_params(p), px_override)
    except Exception as e:
        st.error(f"Could not read image: {type(e).__name__}: {e}")
        return

    if img.px_um:
        src = "manual entry" if px_override else "file metadata"
        extra = f" · objective {img.info.get('objective')}" if img.info.get("objective") else ""
        st.info(f"Scale: **{img.px_um:.4g} µm/pixel** (from {src}){extra}. "
                f"Analysis size: {img.gray.shape[1]}×{img.gray.shape[0]} px.")
    for n in img.notes:
        st.caption("• " + n)

    try:
        mask = oc.load_mask(save_upload(mask_up), img.gray.shape) if mask_up is not None else None
        with st.spinner("Analyzing..."):
            res = oc.analyze(img.gray, img.px_um, p, mask=mask)
            overlay = oc.render_overlay(img.gray, res, img.px_um, title=name)
    except Exception as e:
        st.error(f"Analysis error: {e}")
        return

    for w in res["warnings"]:
        st.warning(w)

    m = res["metrics"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Main path / total", fmt(m["main_path_fraction"]))
    k2.metric("Junctions", fmt(m["n_junctions"]))
    k3.metric("Tortuosity", fmt(m["tortuosity"]))
    k4.metric("RMS deviation / length", fmt(m["rms_dev_rel"]))

    if m["n_junctions"] == 0 and m["n_endpoints"] == 2:
        st.success("Single-tube organoid. Tortuosity and RMS deviation tell you whether the tube is curved or straight.")
    elif m["n_junctions"] > 0:
        st.info("Branched organoid. Here the 'main path' is the longest path through the organoid, so it may turn "
                "into a branch; compare using main path / total, number of junctions and longest side branch, "
                "and do not compare tortuosity with single tubes.")

    a, b = st.columns(2)
    a.image(img.gray, caption="Original image (brightness normalized)", clamp=True)
    b.image(overlay, caption=f"Result · spurs shorter than {res['spur_threshold_um']:.0f} "
                             f"{'µm' if res['units_are_um'] else 'px'} removed")

    st.dataframe(metrics_table(m, res["units_are_um"]))

    stem = os.path.splitext(name)[0]
    row = pd.DataFrame([{"file": name, **m, "spur_threshold_um": res["spur_threshold_um"],
                         "mask_source": "file" if mask is not None else "auto",
                         "notes": " | ".join(img.notes + res["warnings"])}])
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download metrics (.csv)", row.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{stem}_metrics.csv", mime="text/csv")
    d2.download_button("Download result image (.png)", overlay, file_name=f"{stem}_overlay.png", mime="image/png")
    d3.download_button("Download mask (.png) for editing", oc.mask_png(res["mask"]), file_name=f"{stem}_mask.png",
                       mime="image/png",
                       help="Edit in Fiji, then place it next to the original image as <image name>_mask.png; "
                            "folder mode will use this mask automatically.")


# ----------------------------------------------------------------------------
# Tab 2: whole folder
# ----------------------------------------------------------------------------

def tab_batch(p: oc.Params, px_override):
    st.write("Each subfolder is one group, e.g. `data/old_branched/` and `data/new_single_tube/`. "
             "All images use the same parameters currently selected in the sidebar.")
    folder = st.text_input("Image folder path", placeholder=r"D:\organoid\data   or   /Users/name/organoid/data")
    folder = folder.strip().strip('"').strip("'")
    default_out = os.path.join(folder, "results") if folder else ""
    out = st.text_input("Save results to", value=default_out).strip().strip('"').strip("'")
    save_ov = st.checkbox("Save overlay and mask for each image", value=True)

    if folder and os.path.isdir(folder):
        files = bt.find_images(folder, skip_dirs=[out])
        groups = pd.Series([bt.group_of(f, folder) for f in files]).value_counts() if files else pd.Series(dtype=int)
        st.caption(f"Found {len(files)} images: " + ", ".join(f"{g} ({n})" for g, n in groups.items()))
    elif folder:
        st.error("Folder not found.")

    if st.button("Run analysis", type="primary", disabled=not (folder and os.path.isdir(folder))):
        bar = st.progress(0.0)
        status = st.empty()

        def cb(i, n, f):
            bar.progress(min(i / max(n, 1), 1.0))
            status.text(f"{i}/{n}  {f}")

        df, errors = bt.run_batch(folder, p, out or None, px_override, cb, save_overlays=save_ov)
        status.text(f"Done: {len(df)} images succeeded, {len(errors)} failed.")
        st.session_state["batch_result"] = (df, errors, out)

    if "batch_result" not in st.session_state:
        return
    df, errors, out = st.session_state["batch_result"]
    if errors:
        with st.expander(f"{len(errors)} images failed"):
            st.dataframe(pd.DataFrame(errors))
    if df.empty:
        st.warning("No images were analyzed successfully yet.")
        return
    if out:
        st.success(f"Results saved to: {out}")

    if df["group"].nunique() >= 2:
        st.subheader("Comparison between groups")
        st.image(bt.plot_groups(df))
        summ, _ = bt.group_summary(df)
        show = summ.assign(**{c: summ[c].map(fmt) for c in ("median", "q1", "q3")})
        show["p_value"] = summ["p_value"].map(
            lambda v: "–" if v != v else ("< 0.001" if v < 0.001 else f"{v:.3f}"))
        st.dataframe(show[["metric_label", "group", "n", "median", "q1", "q3", "p_value"]]
                     .rename(columns={"metric_label": "Metric", "group": "Group", "median": "Median",
                                      "q1": "Q1", "q3": "Q3", "p_value": "p"}).set_index("Metric"))
    st.subheader("Per-image results")
    st.dataframe(df.set_index("file"))
    st.download_button("Download results table (.csv)", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="results.csv", mime="text/csv")


# ----------------------------------------------------------------------------

def require_password():
    """Stop the page until the user enters the password from .streamlit/secrets.toml."""
    if st.session_state.get("authenticated"):
        return
    try:
        expected = st.secrets["password"]
    except (KeyError, FileNotFoundError):
        st.error("No password is set. Copy .streamlit/secrets.toml.example to .streamlit/secrets.toml "
                 "and set a password.")
        st.stop()
    st.title("Organoid morphology analysis")
    pw = st.text_input("Password", type="password")
    if pw:
        if hmac.compare_digest(pw.encode("utf-8"), str(expected).encode("utf-8")):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()


def main():
    require_password()
    st.title("Organoid morphology analysis")
    st.caption("Single tube or branched · runs entirely on this computer, images are not sent anywhere")
    p, px_override = sidebar_params()
    t1, t2 = st.tabs(["Single image", "Whole folder"])
    with t1:
        tab_single(p, px_override)
    with t2:
        tab_batch(p, px_override)


main()
