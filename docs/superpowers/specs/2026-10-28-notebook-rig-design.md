# Running the CaF lab book on pytweezer: design

## Intent

The CaF side of the experiment is run from Jupyter notebooks in the lab book
(`Box\Working folder\cafmot\datasync\ipython_notebooks\<year>\<month>\<date>.ipynb`,
e.g. `2026\10-Oct\27Oct26.ipynb`) through pycaf. pycaf talks to one MOTMaster
and one locally attached camera. We now need **linked MOTMaster scripts**
(one script driving the CaF MOTMaster on ph-bonesaw and the Rb MOTMaster
together) and **devices on several PCs**, which pytweezer was built for.

The aim is to move these notebooks onto pytweezer devices while disturbing
users' day-to-day workflow as little as possible: the cell that runs a shot
stays roughly one line, images come back to the notebook as numpy arrays, and
the run zips in `mot_master_data` keep the format the existing analysis
(`analysis3.readFromZip`) reads today.

Users are lab members of every level, not only students. They write notebook
cells, not classes.

## How notebooks work today (27Oct26 is representative)

- `expt = pycaf.Experiment(config_path)`, `expt.connect()`.
- A shot: `expt.motmaster_single_run(script, parameter="dummy", value=0.0)`,
  then `get_latest_zip_number()` and an analysis helper that loops
  `CaF.readFromZip(fileno)` → `(images, parameters_dict)`.
- Scans: `expt.scan_wm_laser_set_points(script, "BX", scan_vals)` with
  `motmaster.SaveToggle(True)` and `SetIterations(n)`; the scan leaves one zip
  per point and the user finds them by file-number arithmetic.
- Lock tuning: `MOTAutoTunerWML(expt, script=...)`, configured by attributes
  (`v0_isscan`, `v0_scan_range`, …) and called.
- Camera: `pycaf.thorlabs_camera.ThorlabsGrabber(binx=16, biny=16)` runs free,
  hardware-triggered in bulb mode, and writes one `.tif` per frame into
  MOTMaster's `ExternalFilesPath`; MOTMaster's `StoreRun` sweeps `*.tif` from
  there into the run's zip. `grabber.show_monitor()` opens a Tk live view.
- Every Sep/Oct 2026 notebook (29 of them) reads data with `readFromZip`;
  none use `readFromZipCaFRb`.

## Goals (version one)

1. A notebook on ph-bonesaw can run a linked Rb+CaF shot and get every
   camera's frames back in memory, with the script parameters, in one call.
2. Parameter scans and wavemeter laser set-point scans return images for every
   point and iteration.
3. Auto-tuning of the wavemeter lock set points works through the same rig.
4. Every saved shot is also backed up as today's MOTMaster zip, so
   `CaF.readFromZip(n)` and `get_latest_zip_number()` keep working on new data.
5. The CaF camera's live view is in the pytweezer GUI.

The first live test is: load an Rb MOT (into the tweezers) and a CaF MOT in
one linked script, image the CaF MOT with the CaF ThorCam and the tweezers
with the Rb HamCam.

## Non-goals

- Going through the Experiment Manager queue, or writing `Experiment` classes.
  Notebooks drive devices directly.
- Readouts (molecule number, recapture) in the live view. Image only, for now.
- Making the ThorCam driver's API match the other cameras'.
- Preventing a notebook and a queued experiment from using the same device at
  once. The lab will manage that by hand for now.
- Writing the combined linked `.cs` script (see Dependencies).

## Anti-goals

- **No change to `analysis3.py`** (or anything else in EDMSuite's analysis
  scripts). New zips must read correctly with it as it stands.
- A CaF zip must not contain non-CaF frames: `readFromZip` loads every `.tif`
  in the zip, so mixing species silently corrupts every existing helper.
- No class-writing or queue knowledge required of users for routine work.

## Decisions

### A notebook helper, not experiments (decided by the user)

A small module that notebooks import, sitting where pycaf's `Experiment` sat.
Working shape, names not final:

```python
from pytweezer.notebook import Rig

rig = Rig(motmaster="CaF MotMaster",
          cameras={"CaF": "CaF ThorCam", "Rb": "Rb HamCam"})

shot = rig.run("LinkedRbCaFMOT", dummy=0.0)
images, params = shot.images["CaF"], shot.params

shots = rig.scan_laser("LinkedRbCaFMOT", "BX", scan_vals, iterations=3)
```

- `run` arms every camera, runs the script once (linked or not) and returns a
  shot: frames per camera key, the parameters the script ran with, and the
  zip number(s) it was saved under.
- Per camera, `shot.images[key]` has the same shape and dtype as
  `readFromZip` returns (a float stack in frame order), so existing analysis
  maths applies unchanged; helpers change from looping over file numbers to
  looping over shots.
- Scans (parameter and wavemeter laser set point) return one shot per point and
  iteration.
- Raw `get_device(...)` calls remain available for anything the helper does
  not cover.
- The notebook runs on **ph-bonesaw**, the CaF MOTMaster PC, so it has local
  access to `mot_master_data`.

### Saving (decided by the user)

- MOTMaster's own save stays **on**. MOTMaster writes its zip
  (`parameters.txt`, `hardwareReport.txt`, script copy, …) as today.
- The helper **injects the frames** into that zip from the notebook. This is
  why saving happens at notebook level: frames from cameras on any PC end up
  in the same place.
- **CaF frames** go into MOTMaster's CaF zip
  (`mot_master_data\{yyyy}\{MM}{Mon}\{dd}\CaF{dd}{Mon}{yy}00_{nnn}.zip`).
- **Rb frames** go into a **sibling zip** in the same day folder with the same
  run number, `Rb{dd}{Mon}{yy}00_{nnn}.zip`, with a copy of MOTMaster's
  `parameters.txt`, so `Rb.readFromZip(n)` reads it once `Rb.fileNameString`
  has the Rb prefix.
- Frames must be named so `readFromZip`'s natural sort gives acquisition order,
  e.g. continuing pycaf's `CImage<session>_<counter>.tif` style.
- The camera-side TIFF autosave is not used on this path.

### Which camera images what (decided by the user)

- CaF MOT: the CaF ThorCam on ph-bonesaw (the camera pycaf's
  `ThorlabsGrabber` drives today). It has **no pytweezer device yet**; adding
  one is part of this work.
- Rb: there is no Rb MOT camera. The tweezers are imaged with the **Rb HamCam**
  (`get_device("Rb HamCam")`, ImagEM X2, inside the `Rb Rearrangement Rig`
  composite on PH-BEAST).

### Auto-tune (decided by the user)

`MotMasterInterface.auto_mot` is a straight port of pycaf and may be reshaped
to fit the rig (several cameras, frames collected per shot rather than a fixed
count). Users' calls should look similar to today's `MOTAutoTunerWML` use, not
identical.

### Live monitor (decided by the user)

The CaF ThorCam's live view is the pytweezer GUI image viewer
(`pytweezer/GUI/viewers/image_monitor.py`) on the camera's image stream, not a
window opened by the notebook. No number readouts for now.

### The camera outside notebook runs (decided by the user)

The pytweezer CaF ThorCam device owns the camera; pycaf's `ThorlabsGrabber`
no longer runs alongside it. Runs started outside the notebook (MOTMaster's
own window, run-until-stopped) save CaF zips with no frames, which is already
MOTMaster's default and is acceptable.

### Notebook environment (decided by the user)

Notebooks keep running in the `caftweeznb` uv venv
(`C:\Users\cafmot\.venvs\caftweeznb`), which installs pytweezer by a git pin
to `main`; the pin stays. `get_device` takes device ports from the installed
package's `config.py`, assigned in order, so that copy must match what the
device servers run; adding the `CaF ThorCam` entry shifts the ports of every
later device. Reinstalling pytweezer in `caftweeznb` is therefore a step of
deploying this work (and any later config change).

## Calls I made (check these)

- **Frames per shot are not configured.** Like pycaf's grabber, each camera is
  armed in hardware-triggered mode before `Go()` and, once `Go()` returns, the
  helper collects every frame that arrived. A script giving 3 frames gives 3, a
  script giving 2 gives 2. A missed trigger shows up as a short stack, so the
  helper warns when a camera returns a different count from the previous shot
  of the same script.
- **One shot is one `Go()`, one zip, as today.** MOTMaster runs all its
  iterations inside one `Go()` and saves once, so with `iterations=N` the zip
  holds all N×k frames in order. The helpers' `images[skip*3::3]` striding
  across iterations keeps working, and a scan gives one shot (one zip) per
  point.
- **Saving is a per-call switch, on by default.** Today most notebooks use
  `SaveToggle(False)` for auto-tune and for the "settle on the first scan
  point" step. With save off the rig still returns frames in memory but
  writes no zip and injects nothing; auto-tune defaults to save off, as today.
- The scans are reimplemented in the helper (the existing
  `MotMasterInterface.scan_*` per-point callbacks can't cross sipyco RPC),
  collecting from every camera per point.
- **v1 is plain imaging on the HamCam**: no rearrangement in the linked shot.
  The rig arms the HamCam as the Rb side last configured it; ROI, EM gain and
  sensitivity are set with raw device calls when needed.
- The new CaF ThorCam device ports what the notebooks rely on from
  `ThorlabsGrabber`: serial number from config (the existing `thorcam.py`
  hard-codes the Rb ThorCam's `32148`), bulb exposure, active-high trigger,
  binning, gain (analysis assumes 12 dB), and image broadcast for the
  viewer. The grabber's display features (background subtraction, recapture,
  molecule number) are not ported.
- Camera settings are not recorded in the zip today (pycaf logs them to a
  separate settings log). The helper writes each camera's settings (gain,
  binning, exposure mode, ROI) as a small text file into the zip it injects
  into; `readFromZip` ignores non-`.tif`, non-parameter files.
- **The CaF ThorCam acquires continuously** while its device server runs,
  hardware-triggered, broadcasting every frame to the image stream (so the
  live view works during any run, alignment included) and buffering frames
  with arrival indices. The rig records the buffer position before `Go()` and
  takes every frame after it once `Go()` returns.

## Dependencies

- **The combined linked `.cs` script** (`0. DDSFourSidebands5W_noDDS` +
  `RbTweezerBasic2026_2`) is written separately on the EDMSuite side. The
  helper doesn't depend on what's in a script. It can be built and tested in
  simulation, then on hardware with `LinkedScopeTest` and existing
  single-machine scripts; only the first full Rb+CaF test waits for this
  script.

## Done means

- In simulation and on hardware, `rig.run`, both scans and auto-tune return
  per-camera frames and parameters; `CaF.readFromZip(n)` on the new zip gives
  exactly the CaF frames, and `Rb.readFromZip(n)` on the sibling gives exactly
  the HamCam frames.
- The 27Oct26 workflow (single shots with a recapture analysis, a BX laser
  scan, auto-tune) can be rewritten onto the rig with only the run calls and
  the loop-over-shots change.
- The CaF ThorCam shows live in the pytweezer GUI.
- The linked Rb+CaF first test runs once its script exists.

## Left to the builder

- Module name, class and method names, and the `Shot` object's exact fields.
- How the CaF ThorCam device is structured (extend `thorcam.py` or a separate
  driver) and its CONFIG entry.
- How injection appends to MOTMaster's zip safely (MOTMaster writes it with its
  own zipper at the end of `StoreRun`), and how the helper finds the zip it just
  wrote.
- How "collect every frame that arrived" is done per driver, given the
  ThorCam/ORCA/HamCam APIs differ.
- Tests under `tests/` against the simulated devices, per the `write-tests`
  skill, and a notebook-usage section in the `add-experiment` skill or a new
  skill.
