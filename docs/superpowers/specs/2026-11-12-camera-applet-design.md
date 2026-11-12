# Camera applet: design

Date: 2026-11-12. Status: draft for review.

## Intent

A Camera applet: one window that shows a camera's live image and lets you
change the camera's settings from a sidebar, laid out after Hamamatsu's
HCImage Live (settings panel on the left, image with histogram/levels on the
right). It is for **setting up and aligning between runs**: press Live, tweak
exposure, gain, ROI, look at the image. Today that means a notebook, a
`ContinuousAcquisition` task, or the vendor software; this puts it in the lab's
own GUI, for every camera in CONFIG.

Every setting changed in the applet is also **stored**, so that an experiment
could later choose to pick it up. No experiment does so yet.

## Goals

- One applet that works for any camera in CONFIG; the user picks which.
- The sidebar shows the controls that camera supports, and only those. Settings
  shared by all cameras appear for every camera; settings particular to one
  driver (the ImagEM X2's EM gain, sensitivity, CCD mode) appear only for it.
- Live / Snap, and the image with a levels histogram (auto levels on/off),
  cursor readout, max count, frame number and frame geometry.
- While an experiment task holds the camera, the sidebar is read-only: it shows
  the values the camera has now, greyed, and accepts no edits. The image keeps
  showing whatever the camera broadcasts.
- Every setting the applet applies is written to the saved parameters, one key
  per setting per camera.

## Non-goals

- Changing any experiment to read the stored settings. (`LoadTweezers`,
  `ContinuousAcquisition` etc. keep setting the camera themselves.) An
  experiment that wants one later declares a saved-backed argument, e.g.
  `Number(saved="rb_hamcam_exposure_s")`. Only `Number` and `Integer` take
  `saved=` today, so a choice or on/off setting would be read with
  `get_saved_parameter` in `prepare` instead; extending `saved=` is not part of
  this work.
- A helper that applies all of a camera's stored settings at once.
- Recording or saving frames from the applet. Snap only displays.
- Fighting an experiment for the camera, or warning about one queued.

## Anti-goals (would count as failure even if it works)

- The applet disturbing a running experiment: an edit getting through while a
  task holds the camera, or the applet's polling delaying a shot noticeably.
- A GUI that freezes while a camera call is in flight (a slow DCAM call, a
  device server busy in `wait_for_frame`).
- GUI code that has to change every time a camera is added.
- The sidebar showing a value the camera did not take (e.g. the requested
  exposure rather than the one the driver rounded to).

## Decided (by the user)

- One applet whose sidebar varies with the camera, rather than a subclass per
  camera; the user had no preference and asked for whichever is simpler.
- Layout as in the brainstorm mockup (`layout.html`): sidebar on the left with a
  camera picker and Live/Snap at the top, then collapsible sections; image with
  histogram on the right; status bar under the image.
- Trigger polarity is a shared control, not a ThorCam one: DCAM cameras (ORCA,
  ImagEM X2) have it too.
- Purpose is setup between runs; while a task holds the camera the controls are
  read-only.
- Only Experiment Manager tasks count as holding a camera. A notebook using it
  (`PycafCamera`, `run_local`) is invisible to the applet, and an edit already
  in flight when a task starts may still land; both are accepted, since whoever
  runs the notebook is the person at the applet. No hold mechanism in the
  driver.
- The `CameraSettings` experiments (`pytweezer/experiment/camera_settings.py`,
  one per camera in `pytweezer/experiments/camera_settings.py`) are rebuilt on
  the drivers' control descriptions, so each camera's controls are listed once,
  and what they set is stored in the saved parameters just as the applet's edits
  are.
- The last-written settings are stored so an experiment *could* apply them;
  for now they are only stored.

## Design

### Controls described by the driver

Each camera driver describes its own controls; the applet builds the sidebar
from that description over RPC, so it carries no per-camera code.

- The **description** of a driver's controls is static, declared on the driver
  class (e.g. a class attribute the base defines and each driver extends): one
  entry per control with name, label, section, kind (number / integer / choice
  / bool), unit, range or choices. Static because the `CameraSettings`
  experiments build their form arguments from it at import, with no device
  running; this also lets `simulated_camera_for` copy it onto the simulated
  twin.
- Over RPC, `Camera` gains a method, e.g. `controls() -> list[dict]`, giving
  that description plus each control's **current value**, as read back from the
  camera. ROI is four integer controls (x0, width, y0, height).
- And a method to apply one control, e.g. `apply_control(name, value)`, which
  returns the value the camera actually took (read back).
- Shared controls: exposure, trigger source, trigger polarity, gain, binning,
  ROI. Driver-specific (first cut): ImagEM X2 EM gain on/off, direct EM gain,
  exposure mode (internal/edge/level); ORCA readout speed, exposure mode;
  ThorCam whatever its SDK exposes beyond the shared set. Where a camera cannot
  do a shared control, its driver leaves it out. The ImagEM's shared gain *is*
  its sensitivity (`set_gain` calls `set_sensitivity`): one control, labelled
  Sensitivity, one stored key. The ORCA's gain is its contrast-gain index, the
  ThorCam's is in dB; the label and unit say which.
- On DCAM cameras, exposure mode, trigger source and trigger polarity are
  coupled: `set_dcam_exposure_mode` sets trigger "ext" and forces active-low for
  edge/level. That stays as it is (existing experiments and
  `external_exposure=True` rely on it; *Claude's call*). After any control is
  applied the sidebar re-reads every control, so a coupled change shows at
  once.
- `set_trigger_polarity` becomes a `Camera` method; the DCAM drivers implement
  it through the DCAM trigger-polarity property (the ThorCam already has it).
  `tweezermonitorcam.ThorCam` is a near-copy of `thorcam.ThorCam` and gets the
  same.
- The simulated cameras (`simulated_camera_for`) must describe and accept the
  same controls as their real driver, so the applet works in simulation. This
  does not happen by itself: `simulate()` only stubs methods the simulated class
  lacks, and `SimulatedCamera` inherits `Camera.controls()`, so a driver's own
  list has to be carried over to its twin explicitly.
- The driver keeps the value each setter read back. `controls()` serves those
  kept values rather than re-reading every property from the hardware, so a call
  answered between an experiment's shots costs microseconds. (A call still
  queues behind a `wait_for_frame` in progress; it runs on a worker thread, so
  the applet does not freeze.) *Claude's call.*

### Which cameras are offered

The picker lists every active CONFIG device, including composite sub-devices
such as `Rb HamCam` inside `Rb Rearrangement Rig`, whose driver is a `Camera`.
`camera_names()` in `pytweezer/experiments/camera_continuous.py` computes the
list (inactive ones included); move it somewhere both can import (it is not
experiment code). Inactive cameras and cameras whose server is not running are
listed but say so, with the sidebar empty and the image idle.
The chosen camera is an applet property, so each applet instance remembers its
camera, and two instances can show two cameras.

The image comes from the camera's own image stream (its CONFIG `stream_name`).

### The applet

A new applet under `pytweezer/GUI/viewers/`, subclassing the Image Monitor
(`ImageDisplay` in `image_monitor.py`) as `image_plot_monitor.py` does, so the
image drawing, levels, ROIs, "Copy as camera ROI" and row-major orientation
come for free. Its data sidebar (below the image) stays off by default.

- **Sidebar** (left): camera picker; a status pill (Idle / Live n fps / Held by
  *experiment*); Live (toggle) and Snap; then one collapsible section per
  `section` in the camera's control list. A section of driver-only controls is
  tagged with the camera model.
- **Live** is the camera's own live mode (`start_live` / `stop_live` /
  `is_live`). Experiments already stop live in `prepare` and some restore it in
  `finish` (`LoadTweezers`), so a camera left live by the applet does not break
  them. The rearrangement coordinator (`pytweezer/coordinators/rearrangement.py`)
  does not stop live before acquiring; whether a camera left live breaks it is
  for the builder to check, and if it does, the coordinator should stop live as
  the experiments do. **Snap** acquires one frame with `broadcast=True`; it is
  enabled only when not live and the trigger source is internal, so it never
  waits out a driver timeout (20 s on the ImagEM, which would also stall the
  SLM sharing its process). *Claude's call.*
- **Editing**: a value is applied on Enter / focus-out / selection, never per
  keystroke. The field then shows the value returned by the camera.
  A refused value (driver raises) shows the error on the field and reverts it.
- **ROI** fields x0, width, y0, height; buttons "From drawn ROI" (the region of
  an ROI drawn on the image, via the existing `camera_roi` helper) and "Full
  sensor".
- **No blocking**: every device call runs off the Qt thread; while a call is in
  flight the affected control shows busy, and the rest of the window stays
  responsive.

### Read-only while a task holds the camera

The applet watches which devices the running tasks have reserved
(`ManagerQueue.running_devices()` in `pytweezer/drivers/ni_cards_sources.py`
already does this for the NI Cards driver; or the GUI's `ExperimentFeed`).
When the camera, or the composite it belongs to, is reserved: all controls and
Live/Snap disable, the pill names the experiment, and the sidebar refreshes its
values every few seconds from the driver's kept values (see above), so it shows
what the experiment set without delaying its shots. When the task ends the controls re-enable after one refresh.
If the manager is unreachable, the applet stays editable and says that it
cannot tell whether a task holds the camera.

### Stored settings

Each applied change is written to the saved parameters
(`set_saved_parameters` in `pytweezer/experiment/client.py`) after the camera
has accepted it, storing the value the camera took.

- Keys: `<camera key>_<control name>`, with the camera key a slug of the CONFIG
  name (`rb_hamcam_exposure_s`, `rb_hamcam_sensitivity`, `rb_hamcam_roi`),
  values in SI with the unit recorded. The ROI is four integer keys
  (`rb_hamcam_roi_x0`, `_roi_width`, `_roi_y0`, `_roi_height`), not an array:
  saved arrays are stored as floats, and integer keys can back an `Integer`
  argument, as `LoadTweezers`' `hamcam_roi_*` arguments could later.
- Group: the camera's CONFIG name. Keys are created on first write
  (`create=True`).
- Reason: "Camera applet" (the `CameraSettings` experiments use their own; see
  above); the user is recorded by the client as usual.
- Only edits made in the applet are stored, not values read back while a task
  holds the camera.
- A failure to store (manager unreachable) does not undo the camera change; the
  applet shows that the setting was applied but not stored.

### CameraSettings experiments, rebuilt

`CameraSettings` keeps its job and its form's behaviour (each setting sent only
when ticked or not "unchanged"; camera stopped while changed and returned to
live after; `camera_settings` and `roi_origin` recorded), but its arguments
are generated from the camera's static control description instead of being
written out per driver, and it applies them through `apply_control`, in an order
that respects couplings (binning before ROI, exposure mode before trigger
polarity). The hand-written `HamamatsuSettings`/`ImagEMSettings`/
`ORCASettings`/`ThorCamSettings` go; the per-camera classes in
`pytweezer/experiments/camera_settings.py` keep their names and their
per-camera defaults (e.g. the Rb HamCam's ROI, the CaF ThorCam's 16× binning
and 12 dB). After applying, the values the camera took are written to the
saved parameters with the same keys the applet uses, reason "<experiment
name>, rid <rid>".

Renaming the form fields (they now follow the driver's control names) breaks
saved recipes of these experiments; accepted (*Claude's call*), and the change should
say so.

## Left to the builder

- Exact method names and the shape of a control description.
- Control names for each driver, ranges (from the SDK where it reports them),
  section names and order.
- Widget choices, the pill's wording, the slow refresh period.
- Whether the applet uses `ManagerQueue` or `ExperimentFeed`.
- Where `camera_names()` moves to.
- Updating the `add-applet` and `add-device-driver` skills: a driver now also
  describes its controls, and this applet is an exception to "applets only
  display" (it calls device methods) and to "controls go in the context menu"
  (the sidebar is the point of it). Per CLAUDE.md, the changes to rules in
  those skills need the user's go-ahead.
