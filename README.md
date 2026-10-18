# Organoid morphology analysis: single tube or branched

The app runs entirely on your computer. Drop in images (original `.nd2`, `.tif`, `.png`, `.jpg`) and the app segments the organoid, builds its skeleton, and measures branching and curvature. Images are not sent anywhere.

## 1. Installation (once)

Requires Python 3.10 or later, from python.org (on Windows, tick "Add python.exe to PATH"). Open a terminal in this folder and run:

| Computer | Command |
|---|---|
| **macOS** (Terminal) | `python3 -m pip install -r requirements.txt` |
| **Windows** (Command Prompt) | `python -m pip install -r requirements.txt` |

On Mac, the command always has the 3 (`python3`). Typing `python` or `pip` without the 3 gives `command not found`.

## 2. Opening the app

- **Windows:** double-click `run_app_windows.bat`
- **macOS:** double-click `run_app_mac.command`. The first time you may need to run `chmod +x run_app_mac.command` in Terminal.
- Or type: `python3 -m streamlit run app.py` (Mac) or `python -m streamlit run app.py` (Windows)
- On first run, the terminal may ask for `Email:`; just press Enter to skip.

The app opens in your browser at `http://localhost:8501`. This is a page on your own computer, not on the internet.

**Password:** the app asks for a password. Before the first run, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set your own password in it. `secrets.toml` is not uploaded to GitHub.

## 3. First use: check the scale

Open an `.nd2` file in Fiji and go to **Image → Properties** to see *Pixel width*. Then drop the same file into the app and compare with the line "Scale: … µm/pixel". The two numbers must match. If they don't, or the metadata says exactly 1.0, turn on "Enter scale manually" in the sidebar.

## 4. Recommended workflow

1. **"Single image" tab:** drop in a few representative images from each group. Adjust the parameters in the sidebar until **the yellow outline hugs the organoid edge** and the skeleton looks reasonable.
2. Click **"Save parameters (.json)"**. Every image must be analyzed with **the same parameter set** for the comparison to be fair.
3. **"Whole folder" tab:** paste the path to the data folder, then click **Run analysis**. Each subfolder counts as one group:

```
data/
  old_branched/       a.nd2  b.nd2  ...
  new_single_tube/    c.nd2  d.nd2  ...
```

Results are saved to `data/results/`:

| File | Contents |
|---|---|
| `results.csv` | Metrics for each image (opens in Excel) |
| `group_comparison.csv`, `group_comparison.png` | Median, Q1–Q3 and p-value between groups |
| `overlay/` | Result image for each organoid. **Skim through all of them to catch bad masks** |
| `mask/` | Mask for each image, for fixing if needed |
| `errors.csv` | Images that could not be read or analyzed |
| `parameters.json` | Parameter set used (record it in your methods section) |

## 5. Metrics and how to read them

| CSV column | Meaning | Ideal single tube |
|---|---|---|
| `shape_class`, `is_single_tube` | Verdict per organoid: "single tube" (1) = no closed loops **and** main path ≥ 80% of the skeleton (adjustable); otherwise "branched / network" (0) | single tube / 1 |
| `n_cycles` | Closed loops: empty spaces enclosed by branches, typical of a messy network | 0 |
| `junctions_per_mm` | Branch points per mm of skeleton (does not depend on organoid size) | 0 |
| `main_path_fraction` | Main path length / total skeleton length. **Key metric for "does it form a single tube"** | 1 |
| `n_junctions` | Number of junctions (branch points) | 0 |
| `n_endpoints` | Number of endpoints | 2 |
| `n_branches` | Number of branch segments | 1 |
| `longest_side_um`, `longest_side_rel` | Longest side branch (µm, and relative to the main path) | 0 |
| `tortuosity` | Main path length / straight distance between its two ends | 1 |
| `rms_dev_rel` | RMS deviation from the best-fit straight line, divided by length. This is the "standard deviation from a straight line" | 0 |
| `max_dev_rel` | Maximum deviation, divided by length | 0 |
| `width_mean_um`, `width_cv` | Mean width along the main path and its variability | CV ≈ 0 |
| `solidity` | Area / convex hull area | close to 1 if straight |
| `aspect_ratio`, `circularity` | Length vs width; roundness | |
| `main_length_um`, `skeleton_length_um`, `area_um2`… | Absolute sizes | |

**Important notes for interpretation:**

- **Branching and curvature are two different things.** A curved tube is still one tube (`main_path_fraction` = 1, `n_junctions` = 0) but has high tortuosity. So to compare "branched vs single tube", use `main_path_fraction`, `n_junctions` and `longest_side_rel`.
- **The "main path" is the longest path through the organoid.** In a branched organoid this path may turn from the trunk into a branch, so `tortuosity` and `rms_dev_rel` for the branched group do not describe "trunk curvature". Use these two metrics only to compare curvature **among single-tube organoids** (filter `n_junctions == 0`).
- **`solidity` drops both with branching and with a curved tube**, so use it only as a cross-check.
- **On the overlay**, a single tube shows its main path in cyan. A branched / network organoid has no single main path, so its whole skeleton is drawn in orange, and the title shows the verdict.

**Showing that the new protocol gives single tubes (like a real gut) and the old one does not:**

1. Put old and new images in two subfolders (e.g. `old/`, `new/`) and run the whole folder with **one** saved parameter set.
2. Main result: **% of organoids that are a single tube** per group (bar chart in `group_comparison.png`, Fisher's exact test). For example: "18/20 new vs 2/20 old organoids formed a single tube, p < 0.001".
3. Supporting results, per organoid: `main_path_fraction` (higher = more tube-like), `junctions_per_mm` and `n_cycles` (lower = less messy), with Mann–Whitney tests.
4. Show a few representative overlays from each group as a figure, and skim all overlays to make sure no mask is wrong.
5. Fix the "single tube" threshold (default 0.8) and the "Branch threshold" **before** looking at the results, and report them in the methods.

## 6. Tuning parameters

| What you see on the overlay | Adjust |
|---|---|
| Yellow outline spills into background / includes debris | Increase **Threshold factor** or **Shrink edge inward** |
| Yellow outline falls short / branches missing | Decrease **Threshold factor** or **Shrink edge inward** |
| Mask patchy or broken | Increase **Filter radius** or **Edge smoothing** |
| Thin branches lost | Decrease **Edge smoothing** (it erases structures thinner than about 2× this value) |
| Short orange branches at the edge (fake spurs) | Increase **Branch threshold** |
| Real short branches ignored | Decrease **Branch threshold** |
| **Only one section detected**, the rest is fainter or more transparent | Decrease **Threshold for faint parts** (e.g. 0.4), or decrease **Threshold factor** |
| A **dashed red** outline (dropped part) that is actually a section of the organoid | Increase **Join pieces up to this distance apart** until that section is joined |
| Z-stack file where a section at another depth is lost | Under "Image reading and scale", choose **Merge slices** (default) |
| Outline swallows grainy / cloudy cell clumps around the tubes, many short fake branches, thin connecting tubes missed | Change **Segmentation method** to "Tube network (ridge filter)". If thin tubes are still missed, enter a smaller **Typical tube width**; if the outline is too fat, a larger one |
| Mask bleeds into the background in dark areas | Change **Segmentation method** to "Texture only", or increase **Threshold for faint parts** |
| Fluorescence image analyzes the wrong channel | Under "Image reading and scale", set **Channel** (-1 = auto-pick GFP; channel 2/3 = 1) |

**Branch threshold** is the definition of a "branch" in your study. By default (1.0), a dead-end branch whose length (measured from the junction on the tube axis to the endpoint) is shorter than 1× the mean tube width is treated as a bump, not a branch. In other words, a protrusion sticking out less than about half a tube width from the tube wall is not counted. Agree on this value with your group before running and record it in your methods section.

## 7. When the automatic mask is wrong

1. Download the mask for that image (button in the "Single image" tab, or from `results/mask/`).
2. Fix it in Fiji (paint in or erase; white = organoid, black = background). Masks from ilastik or Cellpose also work.
3. Save it as `<image name>_mask.png` **next to the original image file**, e.g. for `a.nd2` save `a_mask.png`. When running the whole folder, the app uses this mask automatically (column `mask_source` = `file`).

## 8. Running from the command line (no app needed)

```
python3 batch.py path/to/data --params parameters.json
python3 batch.py path/to/data --out results_run2 --px 1.29
```

(On Windows replace `python3` with `python`.)

## 9. Known limitations

- **2D images of 3D objects:** two branches overlapping in the image may be counted as touching, creating fake junctions or closed loops (`n_cycles` > 0). For Z-stack files, the app merges slices by default: each region is taken from its sharpest slice (extended depth of field), so sections at other depths are not lost.
- **Separate pieces:** the app keeps only the main organoid, plus large pieces near it (within "Join pieces"). Dropped large pieces get a dashed red outline on the image and a warning; the `notes` column in the CSV records it too.
- **One organoid per image:** the app keeps the largest region. An organoid touching the image edge gets a warning, since part of it may be cut off.
- **Nearly round organoids** (cyst-like) have no long axis, so the main path metrics are left empty.
- Without a scale, lengths are in pixels. Ratios (fraction, tortuosity, rms_dev_rel, solidity…) can still be compared.
- p-values are only indicative. With small sample sizes, report the median and interquartile range as well, and agree on the statistical method with your supervisor.

## 10. Files

| File | Role |
|---|---|
| `app.py` | User interface (Streamlit) |
| `organoid_core.py` | Image reading, organoid segmentation, skeleton, metrics, overlay drawing |
| `batch.py` | Whole-folder runs, group comparison, plots |
| `.streamlit/config.toml` | Opens the app on this computer only; allows files up to 4 GB |
