# Stage 2: analysis

## Coverage
| stage-1 id | status | covering test(s) |
|---|---|---|
| analysis/analysis_base#1 | not | — |
| analysis/analysis_base#2 | not | — |
| analysis/analysis_base#3 | not | — |
| analysis/analysis_base#4 | not | — |
| analysis/analysis#1 | not | — |
| analysis/analysis#2 | not | — |
| analysis/analysis#3 | not | — |
| analysis/analysis#4 | not | — |
| analysis/analysis#5 | not | — |
| analysis/analysis#6 | not | — |
| analysis/analysis#7 | not | — |
| analysis/analysis#8 | not | — |
| analysis/analysis#9 | not | — |
| analysis/analysis#10 | not | — |
| analysis/data_average#1 | not | — |
| analysis/data_average#2 | not | — |
| analysis/data_average#3 | not | — |
| analysis/experiment_timer#1 | not | — |
| analysis/find_tweezer_atoms#1 | not | — |
| analysis/find_tweezer_atoms#2 | not | — |
| analysis/find_tweezer_atoms#3 | not | — |
| analysis/gaussian_fit#1 | not | — |
| analysis/gaussian_fit#2 | not | — |
| analysis/gaussian_fit#3 | not | — |
| analysis/gaussian_fit#4 | not | — |
| analysis/gaussian_fit#5 | not | — |
| analysis/imageselector#1 | not | — |
| analysis/image_stats#1 | not | — |
| analysis/image_stats#2 | not | — |
| analysis/projections#1 | not | — |
| analysis/projections#2 | not | — |
| analysis/roi_slice#1 | not | — |
| analysis/roi_slice#2 | not | — |
| analysis/roi_slice#3 | not | — |
| analysis/roi_slice#4 | not | — |
| analysis/roi_slice#5 | not | — |
| analysis/rotate_image#1 | not | — |
| analysis/rotate_image#2 | not | — |
| analysis/substract#1 | not | — |
| analysis/subtract_bg#1 | not | — |
| analysis/subtract_bg#2 | not | — |

No test exercises any `pytweezer/analysis/` code for real. The only read-only file, `tests/test_rearrangement.py`, reaches `analysis` solely through `_patch_occupancy`, which replaces `morphological_tophat_high_pass` and `sum_pixel_values` with lambdas and leaves `detector` as `None`, so `SiteScorer.occupancy` (analysis/analysis#1) and `sum_pixel_values` (analysis/analysis#2) are never run. If either broke, no test would fail.

## Existing tests
| test | decision | skill rule that decides it |
|---|---|---|

No test files were assigned for decisions in this area.

## Missed by the review
| test | what it guards | was stage 1 wrong to omit it? |
|---|---|---|
| `tests/test_rearrangement.py::test_arm_streams_sequence_to_slm` | the coordinator's occupancy cut (`pixel_sums > threshold`) becomes the `occ_mask` passed to the phasemask generator, and every streamed frame reaches the SLM | no. The cut and the streaming live in `pytweezer/coordinators/rearrangement.py`, outside this area, and the analysis functions it calls are patched out. But note that the test fails at HEAD with `AttributeError: module 'pytweezer.analysis' has no attribute 'morphological_tophat_high_pass'`. `rearrangement.py` has the same `from pytweezer import analysis as an` import bug that stage 1 flagged only for `transport.py`, and the test's patch targets `pytweezer.analysis.analysis` while the coordinator reads the empty package. Stage 1 should have named `rearrangement.py` as a second affected caller. |
| `tests/test_rearrangement.py::test_arm_reraises_slm_upload_error` | an SLM upload error on the writer thread is raised to the caller | no. This is coordinator behaviour outside this area. It also fails at HEAD, on the same import bug, before it gets to the behaviour it is meant to guard. |

## Verdict
**start again**: this area has no existing tests to rework. 0 kept as is / 0 decided, and none of the 41 stage-1 behaviours is covered or even partly covered. `test_rearrangement.py` touches `analysis` only through monkeypatched stand-ins, so it guards none of the numerical behaviours the skill ranks third, such as site occupancy, box sums, loading statistics, the PSF scores or the Gaussian fits. Those tests have to be written from the stage-1 list, in a new `tests/test_analysis.py` (and per-script files where needed), checked against the independent answers stage 1 gives. Separately, 3 of the 9 tests in `tests/test_rearrangement.py` fail at HEAD. Two hit the analysis import bug above. `test_phasemask_kwargs_merge_config_over_defaults` fails because `Rearrangement` no longer has `phasemask_kwargs`. The coordinators reviewer should pick these up.
