---
name: notebook-rig
description: Save a pytweezer camera's frames into the zips of MOTMaster runs fired from a lab-book Jupyter notebook by pycaf, with PycafCamera (pytweezer/notebook/), keeping them readable by analysis3.readFromZip. Use whenever the user wants a pytweezer camera (Rb HamCam, CaF ThorCam, ORCA) to record during a pycaf scan or motmaster_single_run, asks why a run's zip has no or the wrong frames, or works on pytweezer/notebook/pycaf_camera.py, zips.py, drivers/shot_frames.py or the camera frame-index protocol in camera_base.py.
---

# Camera frames for pycaf notebooks

Lab-book notebooks (`Box\Working folder\cafmot\datasync\ipython_notebooks\...`)
run on **ph-bonesaw** in the `caftweeznb` uv venv
(`C:\Users\cafmot\.venvs\caftweeznb`), which installs pytweezer by a git pin to
`main`. They fire MOTMaster through pycaf's `Experiment`; pytweezer only adds
cameras. Nothing here goes through the queue (that is `add-experiment`), so the
Experiment Manager's device reservations don't cover it: don't run a notebook
scan while a queued task uses the same hardware.

```python
from pytweezer.notebook import PycafCamera

hamcam = PycafCamera(expt, "Rb HamCam")  # key="Rb"
expt.scan_motmaster_parameter(script, "tBlueMOTField", values)
hamcam_images = hamcam.store_images()  # one (frames, H, W) stack per run
hamcam.close()
```

`PycafCamera` puts the camera in external exposure (`external_exposure=True`)
and live mode and wraps `expt`'s `Go`, so every MOTMaster run `expt` makes from
then on (any scan method, or `motmaster_single_run`) is watched.
`store_images()` returns each run's frames since the last call and, with
`save=True`, writes them beside the zip MOTMaster saved for that run; then it
starts afresh. `close()` unwraps `expt` and drops runs not yet stored; the
camera stays live.

## How frames are collected

The camera runs free on hardware triggers in live mode; `start_live` is called
before every watched run (idempotent), so a shared camera another task took out
of live mode comes back. Each run notes the camera's `frames_acquired()` before
`Go`, waits until the count has been still for `frame_settle_s` (up to
`frame_timeout_s`; `drivers/shot_frames.py:settle_frame_counts`), then reads
`read_frames_from(mark)`. Nothing configures frames per run. A camera left on
internal trigger never settles: every run waits `frame_timeout_s` and returns
whatever streamed in.

The camera side is the frame-index protocol on `Camera` (`camera_base.py`):
`start_live`/`stop_live`/`is_live`, `arm`, `frames_acquired`, `read_frames_from`, `settings`.
A pylablib-backed driver gets it for free; others implement
`_acquisition_in_progress`, `_count_frames` and `_read_range`.
`read_frames_from` raises if the ring buffer has already overwritten frames
since the mark; give `PycafCamera(buffer_frames=...)` more. `set_roi` returns the
`(x0, y0)` the camera actually used (kept as `roi_origin`), and every broadcast
frame's header carries it as `roi_x0`/`roi_y0`. Queued tweezer shots also tag
theirs with the task's `rid` (`acquire_n_frames(..., header=...)`); frames
broadcast live during notebook runs carry none, so the live tweezer loading
analysis counts all of them as one run of their own.

## Zips

`pytweezer/notebook/zips.py`. After `Go`, the one new
`{element}{ddMMMyy}{batch}_{nnn}.zip` is found in today's or tomorrow's day folder
under MOTMaster's `data_root_path` (its config file, e.g.
`configuration/caf_mm_config.json`; or `element=`/`data_root=`), and the frames
go in a sibling zip named with `key` in place of the element, e.g.
`Rb27Oct2600_012.zip`, holding `<key>_0000.tif`..., copies of MOTMaster's
`parameters.txt`/`hardwareReport.txt` and `<key>_camera_settings.json` (gain,
binning, ...). Set `Rb.fileNameString = "Rb..."` to read it with
`Rb.readFromZip(n)`. `key` may not be MOTMaster's own element:
`analysis3.readFromZip` loads every `.tif` in a zip, which is why cameras never
share one; `analysis3.py` is deliberately untouched. Runs started outside the
notebook (MOTMaster's own window) get no frames.

## Cameras

- **CaF ThorCam** (ph-bonesaw): bulb trigger, rising edge, 16×16 binning, 12 dB.
  In live mode it broadcasts every frame on `caf_thorcam`. Watch it in the GUI:
  Applet Launcher → image viewer, image stream `caf_thorcam`. The device owns the
  camera, so pycaf's `ThorlabsGrabber` cannot run alongside it.
- **Rb HamCam** (PH-BEAST): starts in external
  exposure mode (`external_exposure` in CONFIG, as do the ORCAs), but keeps
  whatever the Rb side or a `ContinuousAcquisition` task (internal trigger) last
  set. Set ROI, EM gain and exposure mode with an `RbHamCamSettings` task or raw
  calls; DCAM refuses trigger changes while acquiring, so `stop_live()` first
  (`get_device("Rb HamCam").set_exposure_mode("level")`, or `"edge"`/`"internal"`).
  `PycafCamera` puts it back in live mode before each run it watches.

## Deploying a change

`get_device` takes ports from the installed `config.py`, so after a CONFIG
change merges, reinstall pytweezer in `caftweeznb` and restart the device
servers, or notebooks reach the wrong ports.

Tests: `tests/test_notebook_pycaf_camera.py` (fake pycaf experiment and
camera), `tests/test_notebook_zips.py` (a copy of `readFromZip`'s reader proves
compatibility), `tests/test_camera_frames.py`, `tests/test_camera.py`.
