# Rearrangement rig: SLM device + rearrangement coordinator

How the atom-rearrangement system maps onto the composite-device framework
(`docs/device_framework.md`), and a proposed simplification of the ARM pipeline to
do later.

## What this replaced

Two standalone programs, coupled by sockets:

- `pytweezer/rearrangement_node_server.py` — a ZMQ REP server that owned its own
  camera, connected to the SLM through *another* ZMQ socket
  (`communication.SLMClient`), and ran the whole INITIALISE / ARM pipeline inline.
- `pytweezer/rearrangement_node.py` — an `mp.Process` variant of the same idea.
- `pytweezer/drivers/Blink Plus/SLMController/SLMServer.py` — a ZMQ REP server
  wrapping the Meadowlark Blink SDK.

The rearrangement loop's hot path was: GPU computes a phase frame → serialize →
ZMQ → SLM server → hardware, once per interpolation step. That socket hop is
exactly what the composite framework removes.

## What it is now

One composite device, `"Rb Rearrangement Rig"` in `CONFIG["Devices"]`, running one
process on the GPU/SLM host:

```
Rb Rearrangement Rig            (composite: has a "devices" sub-dict)
├── Rb Rearrangement Cam        (class: ...imagemX2:ImagEMX2Camera, role: camera)
├── Rb SLM                      (class: ...slm:SLM,                 role: slm)
└── coordinator                 (coordinator: ...rearrangement:Rearrangement)
```

- **`pytweezer/drivers/slm.py`** — `SLM` wraps the Blink C-wrapper SDK directly as a
  device backend (target `"slm"`); `SimulatedSLM` stands in without hardware. The
  SLM is a first-class device: `get_device("Rb SLM")` works for anyone, unrelated
  to rearrangement.
- **`pytweezer/coordinators/rearrangement.py`** — `Rearrangement(Coordinator)` holds
  direct references to the camera and SLM backends (by **role**, not device name)
  and calls them in-process. `slm.update_mask(frame)` is now a plain method call;
  no frame is ever serialized.

Addressing (see `docs/device_framework.md` for the general rule):

```python
slm   = get_device("Rb SLM")                 # the SLM on its own
coord = get_device("Rb Rearrangement Rig")   # the composite -> its coordinator
coord.initialise(data1, data2, array_shape1, array_shape2, d0, fps, threshold, grid_positions, roi)
img0, img1 = (r := coord.arm_rearrangement())["img0"], r["img1"]
```

`cupy`/`lap` and the GPU math are imported lazily, so the module imports and
`status()`/`test()` work on any machine; `initialise`/`arm_rearrangement` raise a
clear error where the GPU stack is absent.

### Coordinator contract reminders (why it's shaped this way)

- Every coordinator method is a **synchronous** RPC method and stalls the whole
  server (camera, SLM, coordinator) for its duration. `arm_rearrangement` is one
  such call: it blocks until the sequence has played and the reset image is read.
  This is fine for the rearrangement use case (one shot, triggered externally), and
  matches how the old REP server behaved.
- Return values cross a PYON boundary, but they are the *data*, not a status
  wrapper. `arm_rearrangement` returns the `(before, after)` camera frames as a plain
  tuple; `initialise` returns nothing and raises on failure. The old
  `{"status": "success", ...}` dicts were a ZMQ-REP hangover — sipyco propagates
  exceptions natively, so a method returns its payload or raises. The *SLM upload*,
  the latency-critical part, stays in-process regardless.

## The ARM pipeline

`arm_rearrangement` is:

```python
self.slm.update_mask(pm_init_uint8)                      # load initial array
img0 = self.camera.acquire_n_frames(1)[0]                # occupancy image
occ  = an.morphological_tophat_high_pass(img0, 10)       # background subtract
occ_mask = np.fliplr(an.sum_pixel_values(occ, ...)).flatten() > threshold
sequence, _ = PM.generate_rearrangement_sequence(terms1, terms2, occ_mask, d0=d0)
self.slm.run_sequence(sequence, fps=fps)                 # play to SLM
img1 = self.camera.acquire_n_frames(1)[0]                # reset image
return img0, img1
```

The JV pairing, linear interpolation profile, and `fresnel + blaze + zernike`
static background all live in `generate_rearrangement_sequence`
(`pytweezer/phasemask.py`) — verified equivalent to the old inline kernel loop
(same linear profile; `superimpose([pm, fresnel, blaze, zernike])` equals the old
`superimpose([pm, static_background])` because `superimpose` is mod-2π-idempotent).
The occupancy high-pass uses `analysis.morphological_tophat_high_pass` (a
`scipy.ndimage.white_tophat`).

## Remaining optimisation (do later): hardware-triggered playback

`SLM.run_sequence` is **software-timed** — it writes each frame and sleeps `1/fps`.
Since `generate_rearrangement_sequence` produces the whole sequence up front, the
deterministic-timing path is to preload it into the SLM's on-board memory and clock
it out with the SLM's external trigger instead:

```python
self.slm.preload_sequence(sequence)      # DMA all frames to on-board memory
# ... arm the SLM's external trigger; MotMaster/experiment clocks each frame ...
```

`SLM.preload_sequence` already exists (Blink `PreLoad_sequence`, ≤752 frames on the
1024² board). What's missing to complete this path:

1. **A "play preloaded" trigger mode.** The driver needs `SetWaitForTrigger(True)`
   plus a way to advance frames — either the hardware sequencing the preloaded bank
   on each trigger, or a `Select_image(frame)` wrapper. Neither is exposed on `SLM`
   yet; add whichever the Blink SDK supports for preloaded sequences.
2. **Trigger wiring.** Something (MotMaster, a DAQ line) must clock the frames at the
   intended rate; confirm the SLM's `0.696 ms`-after-trigger update timing (manual
   §3.6) fits the rearrangement budget.
3. **Frame-count limit.** Long sequences can exceed 752 frames; chunk or cap `n_steps`
   (tune `d0`) if so.

Until then, `run_sequence` is correct and simple; it just isn't hardware-clocked.
