"""
batch.py
--------
Run the analysis on a whole folder of images. Each first-level subfolder is treated as one group.

    data/
      old_branched/       a.nd2  b.nd2 ...
      new_single_tube/    c.nd2  d.nd2 ...

From the command line:
    python batch.py data --out data/results
    python batch.py data --params parameters.json      (parameter set downloaded from the app)

To use a hand-drawn mask for an image, put a file <image name>_mask.png (or .tif) next to that image.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

import numpy as np
import pandas as pd

import organoid_core as oc

IMAGE_EXTS = (".nd2", ".tif", ".tiff", ".png", ".jpg", ".jpeg")
MASK_SUFFIXES = ("_mask.png", "_mask.tif", "_mask.tiff")

# Group colors in fixed order (checked to be colorblind-safe)
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def _is_mask(name):
    return name.lower().endswith(MASK_SUFFIXES)


def find_images(root, skip_dirs=()):
    skip = {os.path.abspath(d) for d in skip_dirs if d}
    out = []
    for d, subdirs, files in os.walk(root):
        # skip results folders (from this run or earlier runs, including old Vietnamese-named ones) and hidden folders
        subdirs[:] = [s for s in subdirs if os.path.abspath(os.path.join(d, s)) not in skip
                      and not s.startswith(".")
                      and not os.path.exists(os.path.join(d, s, "results.csv"))
                      and not os.path.exists(os.path.join(d, s, "ket_qua.csv"))]
        for f in files:
            if f.lower().endswith(IMAGE_EXTS) and not _is_mask(f) and not f.startswith("."):
                out.append(os.path.join(d, f))
    return sorted(out)


def find_mask(image_path):
    stem = os.path.splitext(image_path)[0]
    for suf in MASK_SUFFIXES:
        if os.path.exists(stem + suf):
            return stem + suf
    return None


def group_of(path, root):
    rel = os.path.relpath(os.path.dirname(path), root)
    return "(root folder)" if rel in (".", "") else rel.split(os.sep)[0]


def _safe_name(rel):
    return rel.replace(os.sep, "__").replace("/", "__")


def run_batch(root, p: oc.Params, out_dir=None, px_um_override=None, progress=None,
              save_overlays=True):
    """Analyze every image in root. Returns (results DataFrame, list of errors)."""
    files = find_images(root, skip_dirs=[out_dir])
    rows, errors = [], []
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if save_overlays:
            os.makedirs(os.path.join(out_dir, "overlay"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "mask"), exist_ok=True)
    for i, f in enumerate(files):
        rel = os.path.relpath(f, root)
        if progress:
            progress(i, len(files), rel)
        try:
            mask_path = find_mask(f)
            img, res = oc.analyze_file(f, p, px_um_override, mask_path)
            row = {"file": rel, "group": group_of(f, root)}
            row.update(res["metrics"])
            row["spur_threshold_um"] = res["spur_threshold_um"]
            row["mask_source"] = "file" if mask_path else "auto"
            row["notes"] = " | ".join(img.notes + res["warnings"])
            rows.append(row)
            if out_dir and save_overlays:
                base = os.path.splitext(_safe_name(rel))[0]
                with open(os.path.join(out_dir, "overlay", base + ".png"), "wb") as fh:
                    fh.write(oc.render_overlay(img.gray, res, img.px_um, title=rel))
                with open(os.path.join(out_dir, "mask", base + "_mask.png"), "wb") as fh:
                    fh.write(oc.mask_png(res["mask"]))
        except Exception as e:  # one failing image does not stop the whole batch
            errors.append({"file": rel, "error": f"{type(e).__name__}: {e}"})
    if progress:
        progress(len(files), len(files), "")
    df = pd.DataFrame(rows)
    if out_dir:
        df.to_csv(os.path.join(out_dir, "results.csv"), index=False, encoding="utf-8-sig")
        if errors:
            pd.DataFrame(errors).to_csv(os.path.join(out_dir, "errors.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(out_dir, "parameters.json"), "w", encoding="utf-8") as fh:
            json.dump({**p.to_dict(), "px_um_override": px_um_override}, fh, ensure_ascii=False, indent=2)
        if len(df) and df["group"].nunique() >= 2:
            summ, _ = group_summary(df)
            summ.to_csv(os.path.join(out_dir, "group_comparison.csv"), index=False, encoding="utf-8-sig")
            with open(os.path.join(out_dir, "group_comparison.png"), "wb") as fh:
                fh.write(plot_groups(df))
    return df, errors


def group_summary(df, metrics=oc.KEY_METRICS):
    """Median, interquartile range per group + p-value (Mann–Whitney for 2 groups, Kruskal–Wallis for >2)."""
    from scipy import stats
    groups = list(dict.fromkeys(df["group"]))
    rows, pvals = [], {}
    for mtr in metrics:
        if mtr not in df:
            continue
        samples = [df.loc[df["group"] == g, mtr].dropna().to_numpy(float) for g in groups]
        p = float("nan")
        valid = [s for s in samples if len(s) >= 2]
        try:
            if len(valid) == 2 and len(valid) == len(samples):
                p = float(stats.mannwhitneyu(valid[0], valid[1], alternative="two-sided").pvalue)
            elif len(valid) >= 3 and len(valid) == len(samples):
                p = float(stats.kruskal(*valid).pvalue)
        except ValueError:  # e.g. all values identical
            p = float("nan")
        pvals[mtr] = p
        for g, s in zip(groups, samples):
            rows.append({
                "metric": mtr,
                "metric_label": oc.METRIC_LABELS.get(mtr, mtr),
                "group": g,
                "n": len(s),
                "median": float(np.median(s)) if len(s) else float("nan"),
                "q1": float(np.percentile(s, 25)) if len(s) else float("nan"),
                "q3": float(np.percentile(s, 75)) if len(s) else float("nan"),
                "p_value": p,
            })
    return pd.DataFrame(rows), pvals


def _fmt_p(p):
    if p != p:
        return "p: could not be computed"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def plot_groups(df, metrics=oc.KEY_METRICS, dpi=110):
    """Small multiples: one panel per metric, each dot = 1 organoid, horizontal bar = median."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [m for m in metrics if m in df]
    groups = list(dict.fromkeys(df["group"]))
    _, pvals = group_summary(df, metrics)
    ncol = 3
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.3 * nrow), dpi=dpi, squeeze=False)
    rng = np.random.default_rng(0)
    ink, muted, grid = "#2b2b29", "#6b6a64", "#e4e3de"
    for k, ax in enumerate(axes.flat):
        if k >= len(metrics):
            ax.set_visible(False)
            continue
        mtr = metrics[k]
        for gi, g in enumerate(groups):
            v = df.loc[df["group"] == g, mtr].dropna().to_numpy(float)
            if not len(v):
                continue
            x = gi + rng.uniform(-0.13, 0.13, len(v))
            ax.scatter(x, v, s=34, color=GROUP_COLORS[gi % len(GROUP_COLORS)],
                       edgecolors="white", linewidths=0.8, zorder=3)
            med = float(np.median(v))
            ax.plot([gi - 0.28, gi + 0.28], [med, med], color=ink, lw=2, zorder=4,
                    solid_capstyle="round")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, fontsize=9, color=ink)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_title(oc.METRIC_LABELS.get(mtr, mtr), fontsize=10.5, color=ink, loc="left", pad=17)
        ax.text(0.0, 1.025, _fmt_p(pvals.get(mtr, float("nan"))), transform=ax.transAxes,
                fontsize=8.5, color=muted, va="bottom", ha="left")
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#bdbcb5")
        ax.tick_params(colors=muted, labelsize=8.5)
    fig.text(0.01, 0.005, "Each dot is 1 organoid · black bar is the median · "
             "p-value: Mann–Whitney (2 groups) or Kruskal–Wallis (>2 groups)",
             fontsize=8.5, color=muted)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Organoid morphology analysis for a whole folder.")
    ap.add_argument("folder", help="Image folder (each subfolder = 1 group)")
    ap.add_argument("--out", default=None, help="Folder to save results (default: <folder>/results)")
    ap.add_argument("--params", default=None, help="Parameter file .json (downloaded from the app)")
    ap.add_argument("--px", type=float, default=None, help="Override µm/pixel (leave empty to read from metadata)")
    ap.add_argument("--no-overlay", action="store_true", help="Do not save overlay images and masks")
    a = ap.parse_args(argv)

    p = oc.Params()
    px = a.px
    if a.params:
        with open(a.params, encoding="utf-8") as fh:
            d = json.load(fh)
        p = oc.Params.from_dict(d)
        px = px or d.get("px_um_override")
    out = a.out or os.path.join(a.folder, "results")

    def prog(i, n, f):
        if f:
            print(f"[{i + 1}/{n}] {f}", flush=True)

    df, errors = run_batch(a.folder, p, out, px, prog, save_overlays=not a.no_overlay)
    print(f"\nDone: {len(df)} images succeeded, {len(errors)} failed. Results in: {out}")
    for e in errors:
        print("  ERROR", e["file"], "→", e["error"])
    return 0 if len(df) else 1


if __name__ == "__main__":
    sys.exit(main())
