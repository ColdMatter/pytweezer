---
name: caf-tweezer-site-analysis
description: Use when asked to analyse a MergeCaFTweezer or MergeCaFTweezerHamCamBackground run (a rid) — whether CaF molecules show up in the tweezers, light or a spot on the HamCam at the tweezer sites, counts or scores per site, a Fresnel lens (focus) scan, tweezer/molecule on-off controls — or when working on the caf_tweezer_sites Results-tab analysis, the site_* arguments or the caf_hamcam_* saved parameters.
---

# Analysing CaF tweezer runs: light at the tweezer sites

A `MergeCaFTweezer…` run images the HamCam once per shot (and, in the
`HamCamBackground` variant, a background frame `HamCamBackgroundGap` later).
The question is almost always "is there light from molecules at the tweezer
sites, and how much". The signal is weak: about 2σ per shot for a single spot,
about 0.5σ per site per shot for a resolved 2×2 array. So the analysis never
searches for spots; it measures at **predicted** positions and judges the
result against the same measurement at empty positions of the same frames.

**Everything the analysis needs is in the run's h5 file and this skill:** the
frames, the arguments, the SLM array and the site calibration. The other
repositories don't come into it. Read the MOTMaster script in EDMSuite
(`MERGE.CaFTweezer…`) only when the user asks about the sequence itself,
such as trigger timing or what a script parameter does.

## Start here

1. Summarise the run and compare it with the previous one, to see what
   changed (spacing, Fresnel lens, ROI, repetitions, scanned argument,
   status). `summarise()` describes one run, so write each run's text to a
   file and `diff` them:

   ```python
   import contextlib, io
   from pytweezer.experiment import Dataset
   text = io.StringIO()
   with contextlib.redirect_stdout(text):
       Dataset(rid, results=[]).summarise()
   ```
2. If the file has what the analysis needs (`missing(m)` is `None`, true for
   files written by commit 81784db onwards, 26 Nov 2026), run it straight from the file:

   ```bash
   uv run python .claude/skills/caf-tweezer-site-analysis/scripts/site_figure.py OUT_DIR RID
   ```
   It prints per-site and combined scores and writes `OUT_DIR/rid<RID>_sites.png`:
   the simulated trap plane, the mean image − background with the sites, the
   array region, the per-shot score histogram, and per-site scores for both
   background rings. In code the same is
   `caf_tweezer_sites(fetch_measurement(rid, results=needed_results(m)), SiteSettings())`
   from `pytweezer.analyses.default.caf_tweezer_sites`.
3. Scanned Fresnel lens: `scripts/fresnel_focus.py OUT_DIR RID` (below).
4. Older files (rid ≤ 1006): see "Files from before 26 Nov 2026".

The Results tab shows the same analysis ("Light at the tweezer sites"). Use
matplotlib figures for reports and the lab book, not GUI screenshots.

## Where the sites are

The sites are predicted from four inputs, all recorded in the file:

| Input | Recorded as | Source |
|---|---|---|
| Trap positions in the focal plane, µm from the blaze spot | constant `slm_arrays` (+ per-point index `slm_array`) | the rig's hologram, written by `SlmArray.update_slm_array` |
| µm → sensor pixels: scale, rotation, x mirrored | arguments `site_scale`, `site_angle` | saved `caf_hamcam_px_per_m` (2.139e6), `caf_hamcam_angle` (−3.0°) |
| Sensor pixel of the blaze spot (array centre) | arguments `site_centre_x`, `site_centre_y` | saved `caf_hamcam_centre_x/_y` |
| ROI origin the camera actually set | constant `hamcam_roi_origin`, `(x0, y0)` | `set_up_hamcam` return value |
| Sensor pixels binned per side | argument `hamcam_binning` (1 when absent) | the form; `set_up_hamcam` fails if the camera won't take it |

The mapping is `geometry_to_pixels` (`analysis/trap_sites.py`). On the HamCam,
2.14 px per µm: a 2 µm array is unresolved (sites ~4 px apart, PSF σ ≈ 3 px),
6 µm gives 12.8 px, 8 µm gives 17 px. Sites closer than two aperture radii (8 px by default) are merged into one.
The **CaF Tweezer Monitor** applet (`GUI/viewers/caf_tweezer_monitor.py`) draws
the same predicted sites live on the HamCam stream, from the rig's current
array and the saved calibration.

- The CaF map is **separate from Rb's `trap_*` calibration**: the CaF imaging
  differs, and Rb's FindTrapSites centre is about 40 px from where the CaF
  spot is. The scale and rotation were taken from Rb's FindTrapSites (run
  896); they reproduced resolved 8 µm and 6 µm arrays.
- Rb's saved `trap_sites` are **full-sensor `(y, x)`**, not `(x, y)`.
- The centre depends on the Fresnel lens: the spot moved ~6 px in x across
  1090–1170 mm. To re-take it, fit the array's spot pattern (fixed relative
  geometry, free centre) to the mean image of a good run at the Fresnel setting
  in use, add the ROI origin, and set the saved parameters (ask the user; they
  are shared lab state). Never fit the centre to the run you are testing.
- **Calibration provenance.** The saved centre (194.7, 304.6) was fitted to
  **rid 1006**, so scoring 1006 with it is circular: use an independent centre
  for 1006. Independent estimates of the blaze-spot sensor pixel agree to about
  ±1 px: rid 993 at 1125 mm (194.7, 306.1); rid 1005's 1134 mm step (195.5,
  305.4); rid 1010 at 1134 mm (194.7, 305.9). A ~1 px error barely changes a
  4 px aperture's score; check the saved parameters' history
  (`saved_parameter_history("caf_hamcam_centre_y")`) for the current
  value's origin.

## How a site is scored

`analysis/site_photometry.py`, on each frame (image − background):

- **Signal** = sum over a disc of 4 px radius − disc area × mean of a ring
  8–12 px out (cancels the uniform offset).
- **Noise** = the same signal on a grid of empty positions of the same frame;
  **z** = (signal − their mean) / their standard deviation.
- **Combined** = sum of the sites' z / √sites, judged against random sets of
  as many empty positions (`combined_chance`). Report the mean ± SE per shot
  and how many of the 2000 random sets reach it.
- Shots with z > 3, against the chance count `bright_by_chance`.

Choices that matter:

- **Spacing ≤ 6 µm: also use the ring 5–8 px.** The 8–12 px ring reaches the
  neighbours' light and understates the sites. Report both.
- **Overlapping sites are one site.** Traps closer than two aperture radii
  (`merge_overlapping`) are scored as a single site at their mean position,
  labelled e.g. `0+1+2+3`: their light can't be told apart. A 2×2 at 2 µm is one
  site at the array centre (z ≈ 2.4 per shot in rid 993 at 1125 mm). Widening
  the aperture can merge the sites of a resolved array.
- **Never pick the aperture position from the data** (a peak search or a fitted
  centre): it biases the score up. If a centre must come from data, fit it on
  half the shots and score the other half.
- **Defocus check:** a larger aperture with its ring just outside it (7 px
  with ring 9–13 px; 10 px with ring 12–16 px). On the 64×80 px window that
  leaves only ~24 (7 px) or ~10 (10 px) empty positions, too few to trust;
  there, compare spot widths from a Gaussian fit instead (as `fresnel_focus.py`
  does). The noise estimate wants `len(score_sites(...).empty)` of about 30 or more.
- **Single shots can't be classified per site** at ~0.5σ. Site-to-site
  correlations near zero are consistent with independent loading, but don't
  show it. Say what signal would be needed (about 4× per shot to separate
  loaded from empty).

## Is the run valid? Check the molecule source

The frame median of image − background is a uniform offset that comes from the
molecule source (probably YAG light in the signal frame), not the tweezers:

| Condition | Spot | Offset (counts/px) |
|---|---|---|
| tweezers on, molecules on | yes | +130 to +190 |
| tweezers off, molecules on (rid 995) | no | +190 |
| tweezers on, molecules off (YAG trigger disconnected, rid 990) | no | −70 to −90 |

Plot the per-shot offset in run order.

- A drop partway through means the source stopped (rid 1003 from shot 74):
  exclude those shots and say so.
- An offset that jumps from shot to shot (rid 1006: −37 to +329) means a
  flickering source. Report the score for all shots, and as a diagnostic also
  split by offset (above and below +100). Don't drop the low shots from the
  headline number.
- A low median offset (+20 to +50, rids 1008 and 1009) means a weak source, so
  a missing spot is inconclusive. `cafmot_counts` does not track it.

## Fresnel lens scans

```bash
uv run python .claude/skills/caf-tweezer-site-analysis/scripts/fresnel_focus.py OUT_DIR RID
```

It fits a Gaussian to the spot in each step's mean image (errors from
resampling the shots) and draws the per-step images above peak brightness and
width against focal length. "Most in focus" is the brightest peak; say
separately if the narrowest width is elsewhere and how significant that is.
26 Nov 2026: brightest at 1131–1134 mm, narrowest at 1145–1167 mm (~2σ).

## Files from before 26 Nov 2026 (rid ≤ 1006)

They lack `slm_arrays`, `hamcam_roi_origin` and the `site_*` arguments.

- **ROI origin**: the camera floors the ROI's start and end to multiples of
  16 px (inferred, consistent across runs): requested x0 164, y0 264 → (160,
  256); requested x0 51, y0 122 → (48, 112).
- **SLM array**: if a later run has the same SLM arguments (`rows`,
  `columns`, `spacing`, `angle`, blaze, Fresnel, pupil, `compensate_envelope`)
  and the same `generator_settings` and `envelope` constants, reuse its
  `slm_arrays`. The trap positions don't depend on the hologram's random
  phases. Otherwise recompute it from the run's arguments and constants on
  PH-BEAST's GPU (cupy; there's no GPU elsewhere). Copy
  `scripts/recompute_slm.py` to `C:\Users\tweez\AppData\Local\Temp\` (never into
  a repo), check nothing is running
  (`ExperimentManagerClient().snapshot()["running"]` is empty; wait otherwise)
  and the GPU is idle (`nvidia-smi`), then
  `ssh beast "cd /d C:\Users\tweez\ColdMatter\pytweezer && .venv\Scripts\python.exe <script> <out_dir> <rid> ..."`
  and `scp` the `.npz` back. It prints recorded vs recomputed uniformity:
  they should agree to ~1e-5. The masks differ pixel by pixel (random
  initial phases), but the trap plane agrees to < 0.5 %.
- Then build the measurement with `dataclasses.replace(m, constants=…,
  arguments=…, results=…, result_shapes=…)`, filling in `slm_arrays` (via
  `slm_tweezer_array.array_summary(hologram, blaze_m)`), `hamcam_roi_origin`,
  the four `site_*` values and `slm_array` zeros, and call `caf_tweezer_sites`.

## Mistakes to avoid

- Running an experiment with `run_local` from a lab-network PC reaches the
  real devices. Use the tests' fakes (`tests/test_slm_tweezer_array.py`,
  `tests/test_tweezer_array.py`) for anything in-process.
- `shoot.py --wait` is in milliseconds.
- `phasemask.py` imports cupy at module level; it can't be imported off
  PH-BEAST.
- Reporting a fitted or searched position's score as significant.
- Leaving the molecule-source check out of a "no spot" conclusion.

## The lab book

Results go in the day's note in CaFMOT-LabBook
(`Daily Notes/YYYY/MM-MMM/DD/`). Read that repo's CLAUDE.md first. Don't
commit in the live vault; Obsidian Git does. Put figures in the day's
`_resources/` with vault-unique names (`YYYY-MM-DD ridNNNN ….png`), and add
each run's `summarise()` text as a collapsed `> [!note]- rid N` callout under
"Dataset summaries". Keep sections concise: one figure that supports the
conclusion, a short table, and a few bullets that say what is and isn't
significant.
