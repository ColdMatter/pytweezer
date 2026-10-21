# Stage 1: Analysis

## pytweezer/analysis/analysis_base.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| analysis/analysis_base#1 | A `(head, data)` returned by `process` is published once on the subclass's own channel: `"_"` for `ImageAnalysis`, `""` for `DataAnalysis` (parametrised over the two). | 4 | unit | quiet: the result goes out on a stream name no viewer or downstream analysis subscribes to, so nothing appears | — |
| analysis/analysis_base#2 | When `process` returns `None`, nothing is published for that message. | 4 | unit | quiet: dropped or unselected frames leak downstream and get averaged or fitted | — |
| analysis/analysis_base#3 | A header-only message (which `recv` returns as a 2-tuple) reaches `process` with `data=None` and does not stop the loop. | 4 | unit | quiet-ish: the process dies on the first such message, and the GUI just shows it as stopped (fails today, see bugs) | — |
| analysis/analysis_base#4 | The client subscribes to the streams in the `imagestreams` / `datastreams` property, the key the Analysis Manager writes the chosen input under. | 4 | unit | quiet: the analysis runs but never gets a message | — |

## pytweezer/analysis/analysis.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| analysis/analysis#1 | `SiteScorer.occupancy` flags exactly the lit sites of a synthetic frame, in sorted-key trap order. The rearrangement coordinator consumes this mask. | 3 | unit | quiet: the rearrangement moves empty traps or misses loaded ones | hand-built frame with bright spots placed at a chosen subset of sites |
| analysis/analysis#2 | `sum_pixel_values` returns each site's `window_size`×`window_size` box sum in `grid_positions` order, clipped at the frame edge. The coordinator fallback and the loading statistics both use it. | 3 | unit | quiet: occupancy and loading probabilities are computed from the wrong pixels | constant frame of value v: interior site gives v·w², corner site gives v·(w//2+1)² |
| analysis/analysis#3 | `detect_trap_sites` on a frame with N separated spots returns N sites, each at its spot's centre of mass rounded to the nearest pixel. | 3 | unit | quiet: every counting window is centred up to a pixel off (fails today, see truncation bug) | spot with weight split 1:99 between two adjacent pixels has COM 0.99 px from the first, so it rounds to the second |
| analysis/analysis#4 | `get_array_loading_statistics` with a fixed threshold returns per-site loading probability equal to the fraction of frames in which that site was lit. | 3 | unit | quiet: reported loading probabilities (the optimisation objective) are wrong | synthetic stack with known per-site occupation pattern, e.g. site k lit in k of 10 frames |
| analysis/analysis#5 | `get_array_loading_statistics` with `threshold_detection=False` reports fidelity near 1 for well-separated background and signal populations. | 3 | unit | quiet: fidelity reported as ~0 (fails today, see bugs) | two populations 100σ apart give an analytic fidelity of 1 − O(e⁻¹²⁵⁰) |
| analysis/analysis#6 | `extract_survival_probability` returns the fraction of sites occupied in the first image that are still occupied in the second, averaged over shots. | 3 | unit | quiet: survival and lifetime numbers are wrong | stacks built so 3 of 4 initially loaded sites survive in every shot give 0.75 |
| analysis/analysis#7 | With `method="psf"`, a site's score does not change when a flat background is added to the frame. | 3 | unit | quiet: background drift shows up as fake photon counts, so atoms are miscounted | weights are zero-mean by construction, so the score difference for frame + c must be 0 |
| analysis/analysis#8 | `detect_loading_threshold` puts the threshold midway between two equal-weight, equal-variance Gaussian populations. | 3 | unit | quiet: the occupied/empty cut is biased, so loading and fidelity are wrong | symmetry: the intersection of identical-width Gaussians is (μ₁+μ₂)/2; use a seeded sample and separation ≫ σ |
| analysis/analysis#9 | `build_psf_templates` recovers a known PSF, normalised to unit sum, from a stack made of that PSF × random occupancy plus a constant background. | 3 | unit | quiet: a distorted template biases every PSF score | known input PSF divided by its sum |
| analysis/analysis#10 | `fit_gaussian` recovers the centre, widths (with `sx` the larger) and amplitude of a noise-free rotated 2-D Gaussian. | 3 | unit | quiet: optimisation scans centre their count windows in the wrong place | analytic Gaussian image with known parameters |

## pytweezer/analysis/crashit.py

No entries: it is a deliberate dummy that prints and exits, used to simulate a crashed analysis.

## pytweezer/analysis/data_average.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| analysis/data_average#1 | In single-experiment mode, after the traces with `_imgindex` 0…n−1 of one shot, it publishes their element-wise mean. | 3 | unit | quiet: a wrong average is published and also written as the background for the subtractor | mean of n known arrays computed by hand |
| analysis/data_average#2 | In continuous mode, it publishes the mean of each consecutive block of `n_average` traces. | 3 | unit | quiet: blocks are mis-sized or mis-normalised (the mode is unreachable today, see bugs) | arithmetic mean of each block of known arrays |
| analysis/data_average#3 | A change in the filter value discards the partial sum, so a published mean only contains traces that share one filter value. | 4 | unit | quiet: averages mix scan points | — |

## pytweezer/analysis/experiment_timer.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| analysis/experiment_timer#1 | After an `Experiment.start` message and then an `Experiment.end` message, it publishes `duration = _endtime − _starttime` with `timestamp = _endtime`. | 4 | unit | quiet: wrong shot durations get plotted or logged | — |

## pytweezer/analysis/find_tweezer_atoms.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| analysis/find_tweezer_atoms#1 | `find_atoms` returns one position per isolated Gaussian spot on a flat pedestal, each within a small sub-pixel tolerance of the true centre. | 3 | unit | quiet: the atom count and positions published to the live display are wrong | synthetic spots at known sub-pixel centres |
| analysis/find_tweezer_atoms#2 | A uniform (constant) frame yields zero atoms. | 4 | unit | quiet: phantom atoms inflate the filling fraction | a constant image has no structure after the high-pass |
| analysis/find_tweezer_atoms#3 | Spots whose raw peak lies below the `threshold` property are not reported. | 4 | unit | quiet: the user's knob has no effect | — |

## pytweezer/analysis/floating_point_arithmetics.py

No entries: nothing in the codebase calls `round_floating_prec`.

## pytweezer/analysis/gaussian_fit.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| analysis/gaussian_fit#1 | With `method="lsq"`, a noise-free 2xN `[x, y]` Gaussian trace on a physical x axis gives back the true amplitude, centre, sigma and offset, with `fwhm = 2√(2 ln 2)·sigma`. | 3 | unit | quiet: wrong widths or positions on the live plot and in anything derived from them | analytic Gaussian with known parameters |
| analysis/gaussian_fit#2 | With `method="moments"`, the centre of a symmetric Gaussian trace is exact. | 3 | unit | quiet: the fast path reports a shifted centre | symmetry: the weighted mean of a symmetric profile is its centre |
| analysis/gaussian_fit#3 | A 1-D input is fitted against the index axis, so `center` comes back in sample units. | 3 | unit | quiet: the centre is reported in the wrong units or frame | Gaussian centred at a known index |
| analysis/gaussian_fit#4 | A flat or empty trace returns `success=False` without raising. | 4 | unit | quiet-ish: either the process dies or a garbage fit is flagged as successful | — |
| analysis/gaussian_fit#5 | The published header keeps the incoming header keys (e.g. `timestamp`, `_imgindex`) alongside the fit parameters, and carries no array. | 4 | unit | quiet: viewers cannot line the fit up with its shot | — |

## pytweezer/analysis/imageselector.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| analysis/imageselector#1 | Only the frame whose `_imgindex` equals `SelectedImage` is republished, unchanged, on channel `"_"`. | 4 | unit | quiet: downstream analyses process the wrong image of the shot | — |

## pytweezer/analysis/image_stats.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| analysis/image_stats#1 | For a known frame, it publishes `n`, `mean`, `std`, `stderr = std/√n`, `min`, `max` and `sum`. | 3 | unit | quiet: wrong numbers in the viewer's data sidebar | stats of a small hand-written integer frame |
| analysis/image_stats#2 | `_imgindex` and `timestamp` are copied from the image header into the stats message when present. | 4 | unit | quiet: stats cannot be matched to their frame | — |

## pytweezer/analysis/__init__.py

No entries: it is an empty package marker.

## pytweezer/analysis/projections.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| analysis/projections#1 | `_x` carries the column sums (intensity vs x) and `_y` the row sums, each as a 2xN `[coords, values]` array. | 3 | unit | quiet: x and y profiles are swapped or transposed, so fitted widths go to the wrong axis | a non-square, asymmetric frame whose row and column sums are computed by hand |
| analysis/projections#2 | Profile coordinates are `(index + _offset)·_imgresolution`, so a crop from `roi_slice` maps back to the parent frame's physical axis. | 3 | unit | quiet: fitted centres are offset or wrongly scaled | a crop with known offset and resolution has its first coordinate at `offset·res` |

## pytweezer/analysis/random_number.py

No entries: it is a test stub that publishes random integers, and nothing relies on its output.

## pytweezer/analysis/roi_slice.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| analysis/roi_slice#1 | An ROI given in physical units (with the header's `_offset` and `_imgresolution`) publishes exactly the parent pixels underneath it. | 3 | unit | quiet: downstream analyses see the wrong region | invert `physical = (pixel + offset)·res` by hand for a chosen ROI |
| analysis/roi_slice#2 | The crop's header `_offset` is the parent offset plus `(j0, i0)`, so the crop's pixel (0, 0) maps to the same physical point as parent pixel `(i0, j0)`. | 3 | unit | quiet: crops are drawn and projected at the wrong place | round-trip of one pixel's physical coordinate |
| analysis/roi_slice#3 | An ROI dragged inverted (negative size) gives the same crop as its normal equivalent. | 4 | unit | quiet: an empty or wrong crop | symmetry: pos+size with size<0 is the same rectangle |
| analysis/roi_slice#4 | An ROI that lies wholly outside the frame publishes nothing. | 4 | unit | quiet: empty arrays get published downstream | — |
| analysis/roi_slice#5 | Each ROI is published on channel `_<last component of its path>`. | 4 | unit | quiet: subscribers to `<name>_<roi>` get nothing | — |

## pytweezer/analysis/rotate_image.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| analysis/rotate_image#1 | At `angle = 0`, both the image and the header `_offset` pass through unchanged. | 3 | unit | quiet: rotated frames are placed at the wrong physical origin (fails today, see bugs) | identity transform |
| analysis/rotate_image#2 | `_offset` is rotated by the same angle and in the same sense as the image, so its length is preserved. | 3 | unit | quiet: positions in rotated frames are mis-located | a rotation preserves the norm; 90° maps (x, y) to (−y, x) or (y, −x) depending on the image convention |

## pytweezer/analysis/substract.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| analysis/substract#1 | The background's values row is subtracted from the incoming 2xN trace's values row, and the coordinate row is left untouched. | 3 | unit | quiet: a background-corrected trace with the wrong baseline or a shifted axis (crashes today, see bugs) | difference of two hand-written arrays |

## pytweezer/analysis/subtract_bg.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| analysis/subtract_bg#1 | Given a 16-bit background and an 8-bit image, each published pixel is `|img − bg / 2⁸|`. | 3 | unit | quiet: a wrongly scaled background leaves a residual on every frame | hand arithmetic on a tiny background PNG written to `tmp_path` |
| analysis/subtract_bg#2 | A frame whose shape is the transpose of the background's is corrected against the transposed background. | 4 | unit | quiet-ish: the subtraction is misaligned, or the process crashes on a shape mismatch | — |

## Suspected bugs

- pytweezer/analysis/analysis.py: `detect_trap_sites` and `detect_trap_sites_grid` start the threshold at `img.max()`, a NumPy scalar of the image dtype. On a `uint16` frame (raw, or after `white_tophat`, which keeps the dtype), `threshold -= 100` wraps around (to 65486 for 50 − 100) and never reaches `< 1`, so the loop spins forever when the requested count is never matched. When it does exit on `< 1`, it returns the last centres, which are the wrong count, without raising. It also raises `NameError` if `img.max() < 1`. Smallest input: `detect_trap_sites(np.array([[0, 50]], dtype=np.uint16), atom_number=2)` (hangs).
- pytweezer/analysis/analysis.py: `detect_bright_points` truncates each centre of mass with `.astype(int)` instead of rounding, so every site is biased by up to one pixel towards the origin. Smallest input: `detect_bright_points(np.array([[0, 1, 99]]), threshold=0)` gives `[[0, 1]]`, where the true COM is 1.99.
- pytweezer/analysis/analysis.py: in `get_array_loading_statistics` and `_grid`, the fixed-threshold branches (and the fallback after GMM failure) compute `fidelity = P(false negative) + P(false positive)`. That is the unweighted total error, not the fidelity. Smallest input: photons `[-1, 1, 99, 101]` with `threshold_detection=False, threshold=50` give fidelity ≈ 0 instead of ≈ 1. Separately, the message says "Setting fidelity to 0" but the code sets NaN.
- pytweezer/analysis/analysis.py: `get_array_loading_statistics_grid` passes `grid_shape` positionally into `sum_pixel_values(image, grid_positions, grid_shape, window_size=...)` and into `SiteScorer.from_images(images, grid_positions, grid_shape, window_size=...)`, so both paths raise `TypeError: got multiple values for argument 'window_size'`. Its later `photon_array[:, n, m]` indexing also assumes a 3-D array that neither scorer returns. Smallest input: `get_array_loading_statistics_grid([np.zeros((10, 10))], {(0, 0): (5, 5)}, (1, 1))`.
- pytweezer/analysis/analysis.py: `extract_temperature` drops the NaNs from `survival_prob_list` first, then builds the mask for `dropt` from the already-filtered list, so the lengths and the pairing no longer match. Smallest input: `extract_temperature(np.array([1., 2., 3.]), [0.9, np.nan, 0.5], plot=False)` raises `IndexError`. Its bare `except:` also returns a fabricated 100 µK when the fit fails, and with `plot=True` it then raises `NameError` on `params`.
- pytweezer/analysis/analysis.py: `fit_gaussian` sorts `sx` and `sy` so that `sx` is the larger width, but leaves `sx_err`/`sy_err` and `theta` unswapped. The errors then belong to the other axis. The sort also means `extract_cloud_temperature`'s "Tx"/"Ty" are really major-axis/minor-axis temperatures. Smallest input: a Gaussian image with fitted `sx=5`, `sy=20` returns `sx=20` with the 5-px width's error.
- pytweezer/analysis/analysis.py: `index_grid_positions` converts pixels to µm and back, then uses `.astype("int64")`, which truncates float round-off and moves about 4 % of sites one pixel down. Smallest input: x pixel 1 with `x_centre = 10.5`, since `int((1 - 10.5) * -hamcam_um_per_px * -hamcam_px_per_um + 10.5)` gives `0` (value 0.9999999999999982).
- pytweezer/analysis/analysis_base.py: `run()` unpacks `msgstr, head, data = msg`, but `GenericClient.recv` returns a 2-tuple for header-only messages, so the first such message raises `ValueError` and kills the process. `gaussian_fit.py`'s `if data is None` branch can therefore never run. Smallest input: a client whose `recv()` returns `("x", {"a": 1})`.
- pytweezer/analysis/data_average.py: `_single_experiment` defaults to the string `"True"` and is tested for truthiness, so any non-empty value, including `"False"`, selects single-experiment mode and the continuous branch is unreachable. Smallest input: property `_single_experiment = "False"`. Also:
  - After a mid-shot reset (filter change), the partial sum is still divided by `n_average`. With `n_average=2` and the filter value changing at `_imgindex 1`, it publishes `data1 / 2`.
  - For integer-dtype traces, `average /= n_average` raises `UFuncTypeError` outside the `try`.
  - `n_average` is read once, so live retuning is ignored.
  - Every instance writes to the hard-coded `/Analysis/Data/axcol_bg_substract/background`.
- pytweezer/analysis/experiment_timer.py: an `Experiment.end` with no preceding start (`starttime == 0`) satisfies `0 < endtime` and publishes the absolute end time as the duration. Smallest input: one end message with `_endtime = 100` and no start gives `duration = 100`.
- pytweezer/analysis/floating_point_arithmetics.py: `round_floating_prec` rounds to `dig − 3` = 12 decimal places, not significant figures, so small values are zeroed. Smallest input: `round_floating_prec(1e-13)` gives `0.0`. It is currently unused.
- pytweezer/analysis/random_number.py: `self.dataq.send(data, "_n")` passes `"_n"` as the array argument `A`, so the number is published on `<name>`, not `<name>_n`. Smallest input: any `Experiment.start` message.
- pytweezer/analysis/rotate_image.py: the `_offset` "rotation" has no minus sign. It swaps x and y at 0° and is the identity at 90°. It also mutates the incoming header in place. Smallest input: `angle = 0`, `_offset = [10, 0]` gives `[0, 10]`.
- pytweezer/analysis/substract.py: `np.float` was removed in NumPy 1.24 (2.4.4 is installed), so the first loop iteration raises `AttributeError` and the process dies. The default background `"[]"` is a string, which `np.array(..., dtype=float)` also rejects. Smallest input: `np.array("[]", dtype=np.float)`.
- pytweezer/analysis/subtract_bg.py (possibly intended): `np.abs(img - background)` rectifies negative residuals, so background-free regions have a positive mean and summed counts are biased upwards. Smallest input: equal bit depths, `img = [[0]]`, `bg = [[1]]` gives `1` instead of `-1`.
- pytweezer/coordinators/transport.py (caller of this area, for the coordinators reviewer): `from pytweezer import analysis as an` resolves to the empty package `pytweezer/analysis/__init__.py`, not to the stale duplicate module `pytweezer/analysis.py`, because a package takes precedence over a module of the same name. The no-detector occupancy path therefore raises `AttributeError` on `an.morphological_tophat_high_pass` / `an.sum_pixel_values`. Smallest input: `_extract_occupancy(image, threshold)` with no `detector` in `_state`.
