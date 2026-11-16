# Live mode as the cameras' standard operating mode

## Intent

Cameras stay live (continuous ring-buffer acquisition, every frame broadcast
on the camera's stream) all the time, MOTMaster experiments included. An
experiment no longer switches a camera into snap mode and back; it takes its
frames out of the running acquisition. The Camera applet then shows every
frame an experiment takes, as it arrives, and cameras are no longer stopped
and restarted by every task. `ContinuousAcquisition` is removed: the Camera
applet and the `CameraSettings` experiments have replaced it.

## Constraint: shot tags on the live stream

Analyses tell a task's frames apart by their broadcast header: `rid` (which
task) and `index` (which frame of the shot: 0, 1, ...). `TweezerLoading`
scores the frame whose `index` is its `image_index`; `ShotBackground` and the
tweezer loading plots group by `rid`. Live broadcasts carried neither (`index`
was the frame's number in the acquisition). Experiments used to broadcast
their own frames with both.

## Approaches

- **Experiment hands the camera the shot's tag (chosen, the user's
  proposal).** At the start of each shot the experiment tells the driver "a
  shot starts now, tag it `{"rid": ...}`"; the driver notes the frame count
  as the shot's mark and the live thread tags the frames from that mark on.
  The live stream stays the one place frames are broadcast.
- Pause the live broadcast while a task holds the camera and let the
  experiment broadcast its frames, as now. Rejected: the applet goes blank
  mid-task, which is what live mode is meant to end.
- The driver asks the Experiment Manager which task holds it. Rejected: ties
  drivers to the queue, and gives no shot-relative `index`.

## Design

**Driver (`Camera` in `drivers/camera_base.py`)**

- `start_shot(header) -> int`: the current frame count becomes the shot's
  mark; returns it. Frames numbered from the mark on are broadcast with
  `header` added and `index` = frame number − mark.
- `end_shots()`: frames from now on are broadcast untagged again (plain
  running `index`, no `rid`).
- The driver keeps each (mark, header) until the live thread has broadcast
  past it, so the last frames of one shot keep their tag when the next shot
  starts before they were broadcast. Starting a new acquisition clears them.
- `wait_frames(mark, n, timeout=None) -> ndarray`: block until frames
  `mark .. mark+n-1` exist, return them; `timeout` defaults to the camera's
  `timeout`. Raises `TimeoutError` if they do not arrive.

**Experiments**

- `LoadTweezers` (and through it `TweezerArrayMeasurement`, `FindTrapSites`):
  `prepare` stops live, applies its settings (DCAM refuses trigger changes
  while capturing), then starts live; no more `setup_acquisition(mode="snap")`.
  `take_shot`: `mark = hamcam.start_shot({"rid": rid})`, run the sequence,
  `hamcam.wait_frames(mark, 2)`. A gain change between shots restarts live
  around `set_gain`. It no longer broadcasts frames itself.
- `LoadTweezersNImages`: the same with 10 frames. The `hamcam_n_images`
  stream goes (nothing subscribes to it); its frames are on the HamCam's
  stream, tagged.
- `LoadTweezersMOTImage`: the ORCA likewise — live from `prepare`,
  `start_shot` before the HamCam's shot, `wait_frames(mark, 1)` after.
- `RbMotMasterRunner`: takes its marks with `start_shot({"rid": rid})` so its
  frames are tagged; keeps waiting for the frame counts to settle, since it
  does not know how many frames a script takes.
- `finish`: `end_shots()`; the camera is left live with the experiment's
  settings (external trigger). The "restart live only if it was live" logic
  goes.

**Removed:** `experiments/camera_continuous.py`, `tests/test_camera_continuous.py`,
the `ContinuousAcquisition` line in the `notebook-rig` skill.

## Non-goals

- The rearrangement coordinator (sub-millisecond timed path; stays on its own
  acquisitions), `helper_scripts/optimisation.py`, `PycafCamera`, the
  applet's Snap.
- Restoring a camera's earlier trigger after a task.

## Left to the builder

- Existing tests whose fakes model snap acquisition (`test_tweezer_array.py`
  and the like) follow the design: update their fakes to `start_shot` /
  `wait_frames`. No new tests (standing rule).
- Skills that describe the experiments' acquisition (`add-experiment`,
  `add-device-driver`, `notebook-rig`) are brought up to date in the same
  change.
