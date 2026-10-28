---
name: notebook-rig
description: Run MOTMaster shots, parameter or laser scans and MOT auto-tuning from a lab-book Jupyter notebook through pytweezer's Rig (pytweezer/notebook/), getting every camera's frames back as numpy arrays and keeping MOTMaster's run zips readable by analysis3.readFromZip. Use whenever the user is moving a pycaf notebook (Experiment, motmaster_single_run, scan_wm_laser_set_points, MOTAutoTunerWML, ThorlabsGrabber, get_latest_zip_number) onto pytweezer, writing or debugging notebook cells that run the CaF/Rb experiment directly (not through the queue), asks why a shot's zip has no or the wrong frames, or works on pytweezer/notebook/rig.py, zips.py or the camera frame-index protocol in camera_base.py.
---

# Running the experiment from a notebook

Lab-book notebooks (`Box\Working folder\cafmot\datasync\ipython_notebooks\...`)
run on **ph-bonesaw** in the `caftweeznb` uv venv
(`C:\Users\cafmot\.venvs\caftweeznb`), which installs pytweezer by a git pin to
`main`. They drive devices directly with `Rig` — no queue, no `Experiment`
classes (those are `add-experiment`).

```python
import numpy as np
from pytweezer.notebook import Rig

rig = Rig("CaF MotMaster", {"CaF": "CaF ThorCam", "Rb": "Rb HamCam"})
rig.linked = True                        # for a linked Rb+CaF script

shot = rig.run("LinkedRbCaFMOT", iterations=3, dummy=0.0)
images, params = shot.images["CaF"], shot.params   # what CaF.readFromZip(n) gave
shot.zip_number                          # MOTMaster's run number, None if save=False

shots = rig.scan("BX_BMOT_OP_magtrap", "tHold", [0, 10, 20])
sp = rig.laser_set_points()
shots = rig.scan_laser("BX_BMOT_OP_magtrap", "BX", scan_vals, iterations=3, settle_s=10)
results = rig.auto_mot("0. DDSFourSidebands5W_noDDS",
                       {"v0": np.linspace(sp["v0"] - 10e-6, sp["v0"] + 10e-6, 30)},
                       iterations=3, display=True)
rig.close()                              # releases the link
```

Moving a pycaf helper over: replace the `for fileno in range(file_start, file_end + 1):
images, parameters_dict = CaF.readFromZip(fileno)` loop with
`for shot in shots: images, parameters_dict = shot.images["CaF"], shot.params`.
The maths (`images[skip*3::3]` across iterations included) is unchanged: one
shot is one `Go` and holds all its iterations' frames in order. An interrupted
scan leaves its finished shots in `rig.last_shots`.

## How frames are collected

Cameras are armed once (`Camera.arm`) and run free on hardware triggers. `run`
notes each camera's `frames_acquired()` before `Go`, waits until every count has
been still for `frame_settle_s` (up to `frame_timeout_s`), then reads
`read_frames_from(mark)`. Nothing configures frames per shot; a count that
differs from the previous shot of the same script raises a `UserWarning`
(usually a missed trigger). A camera left on internal trigger never settles:
every shot waits `frame_timeout_s` and returns whatever streamed in. Simulated
cameras free-run the same way.

The camera side is the frame-index protocol on `Camera` (`camera_base.py`):
`arm`, `frames_acquired`, `read_frames_from`, `settings`, `start_live`/`stop_live`.
A pylablib-backed driver gets it for free; others implement
`_acquisition_in_progress`, `_count_frames` and `_read_range`.

## Zips

MOTMaster's save stays on (`save=True` default; `auto_mot` defaults to
`save=False`). After `Go` the rig finds the one new `CaF{ddMMMyy}{batch}_{nnn}.zip`
in today's or tomorrow's day folder under `data_root_path`
(`configuration/caf_mm_config.json`) and:

- the camera whose key equals MOTMaster's element (`"CaF"`) is appended to that
  zip as `CaF_0000.tif`...;
- every other key gets a sibling zip, e.g. `Rb27Oct2600_012.zip`, with copies of
  MOTMaster's `parameters.txt`/`hardwareReport.txt`, so set
  `Rb.fileNameString = "Rb..."` to read it with `Rb.readFromZip(n)`;
- each zip also gets `<key>_camera_settings.json` (gain, binning, ...).

`analysis3.readFromZip` loads every `.tif` in a zip, which is why species never
share one; `analysis3.py` is deliberately untouched. Runs started outside the
notebook (MOTMaster's own window) save zips with no frames.

## Cameras

- **CaF ThorCam** (ph-bonesaw): bulb trigger, rising edge, 16×16 binning, 12 dB,
  `live: True` — it broadcasts every frame on `caf_thorcam`. Watch it in the GUI:
  Applet Launcher → image viewer, image stream `caf_thorcam`. The device owns the
  camera, so pycaf's `ThorlabsGrabber` cannot run alongside it.
- **Rb HamCam** (PH-BEAST, inside `Rb Rearrangement Rig`): armed as the Rb side
  last left it; set ROI, EM gain and trigger with raw calls
  (`get_device("Rb HamCam").set_external_exposure_mode()`). Not usable from the
  rig while a rearrangement coordinator owns it.

## Deploying a change

`get_device` takes ports from the installed `config.py`, so after a CONFIG
change merges, reinstall pytweezer in `caftweeznb` and restart the device
servers, or notebooks reach the wrong ports.

Tests: `tests/test_notebook_rig.py` (fake MOTMaster and cameras),
`tests/test_notebook_zips.py` (a copy of `readFromZip`'s reader proves
compatibility), `tests/test_camera_frames.py`.
