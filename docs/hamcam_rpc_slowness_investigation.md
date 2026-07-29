# HamCam RPC slowness — investigation notes (2026-07-28)

Status: a real O(n^2) bug in sipyco was found and fixed on branch `sipyco-fix`
(commit `5aef1f6`), **not merged, not pushed**.

**Whether it fixes the reported symptom is still unmeasured.** The one attempt
to test against real hardware was run over a Tailscale DERP relay and is
invalid — see "Hardware results" below. A valid test has to run on the lab LAN.

## Symptom

Images from the HamCam (`ImagEMX2Camera`, served via `get_device("Rb HamCam")` /
`get_device("CaF HamCam")`) take a very long time to come back over RPC —
bad enough to notice, worse the more frames requested in one call
(`acquire_n_frames(n)`).

## Root cause

Not the camera, not the network, not PYON's base64 encoding. It's a bug in
`sipyco.pc_rpc._socket_readline` (vendored dependency,
`.venv/lib/python3.11/site-packages/sipyco/pc_rpc.py:57`), which every
synchronous `sipyco.pc_rpc.Client` reply goes through:

```python
def _socket_readline(socket, bufsize=4096):
    buf = socket.recv(bufsize).decode()
    offset = 0
    while buf.find("\n", offset) == -1:
        more = socket.recv(bufsize)
        ...
        buf += more.decode()
        offset += len(more)
    return buf
```

`buf += more.decode()` is only O(1) amortized when CPython's in-place
`PyUnicode_Append` optimization fires, which requires the string's refcount to
be exactly 2 at the point of the append. Here `buf.find(...)` sits in the
`while` **condition**, which is evaluated (and holds a reference to `buf`)
before the body runs — that extra reference is enough to defeat the in-place
optimization on every iteration. Confirmed by instrumenting: **1998
reallocations out of 1999 concatenations**, vs. **8** when the same `find`
call is moved into the loop body instead of the condition.

Net effect: reading a reply is **O(n²)** in the reply size, because every 4 KB
chunk copies the entire reply accumulated so far. A frame is exactly the kind
of payload that hits this — anything under ~1 MB is invisible, multi-frame
`acquire_n_frames(n)` grabs are where it blows up.

For an 8 MB reply, time breakdown was: 4430 ms in the readline loop, 50 ms
actually in `recv()`, 17 ms in `pyon.decode`. The bug is entirely in that one
function.

Only the **synchronous** `Client` is affected. `AsyncioClient` already reads
through asyncio's C-implemented `readline` (opened with
`limit=100*1024*1024`), which doesn't have this shape. `get_device()` (the
one used everywhere in pytweezer, including notebooks) returns a sync
`Client`, so this hits real usage directly.

## Measurements

Real sipyco server on loopback, 512×512 uint16 frames, fresh Python process
per measurement (rules out interpreter/JIT warm-up smearing results between
cases). Reproducible with `tools/bench_sipyco_readline.py` (see below).

| frames | raw size | stock sipyco | patched | speedup |
|---|---|---|---|---|
| 1 | 0.5 MB | 16–29 ms | 4–8 ms | ~4× |
| 10 | 5.2 MB | 1.7–1.9 s | 43–59 ms | ~38× |
| 50 | 26.2 MB | 48.7–54.8 s | 240–371 ms | ~200× |
| 100 | 52.4 MB | **~198–207 s** | **0.48–0.70 s** | ~350–420× |

(`pybase64` was added to `pyproject.toml` after the numbers above were first
taken; both columns already reflect it — see below for what it specifically
buys.)

### Important caveat on those numbers: they are *first-call* figures

Each measurement above is one grab in a fresh process. That turns out to be the
stock reader's **worst case**, not its typical case. Timing six consecutive
10-frame grabs in one process (synthetic server, loopback, ms per call):

```
stock     1558.7   425.5   409.0   425.7   416.0    25.9
patched     46.2    32.8    24.2    30.5    30.1    34.8
```

The stock reader is ~34x slower on the first grab but settles to ~13x, because
whether `buf += chunk` costs a full copy depends on the allocator being able to
grow the buffer in place — which it increasingly can once a process has settled
into reusing one arena for same-sized replies. So the honest summary is:

* the defect is real and always costs something (patched was never slower);
* the penalty is largest on the first big grab in a process and on a busy or
  fragmented heap, and shrinks — sometimes to nothing — on repeated same-sized
  grabs in a quiet process;
* the headline "~418x" is a cold-process number and should **not** be quoted as
  the expected steady-state gain.

Corollary for measuring this: a benchmark that discards a warm-up and averages
the rest can show the fix doing *nothing*, and one that times only a cold first
call shows the maximum. Report the per-call curve, not a single average.

This matches the real `hybrid_exp_test.ipynb` / `28Jul26Exp.ipynb` /
`rearrangement_server_test.ipynb` usage pattern directly — e.g.
`imgs_ham = hamcam.acquire_n_frames(n_iterations)` and
`local_cam.acquire_n_frames(n_iterations * 2)` — any loop that grabs more
than a handful of frames per RPC call pays this.

## Hardware results (2026-07-28, `Rb HamCam` on PH-BEAST) — INVALID, do not use

An attempt to measure this against the real camera from a laptop produced these
numbers, which are **an artefact of the network path and say nothing about the
camera or the fix**:

| frames | MB | stock (med) | patched (med) | ratio |
|---|---|---|---|---|
| 1 | 0.5 | 1241 ms | 1402 ms | 0.9x |
| 10 | 5.2 | 8381 ms | 9290 ms | 0.9x |

Why they are invalid: the laptop reaches PH-BEAST over **Tailscale, relayed via
a DERP server in London** (`tailscale ping` reports "direct connection not
established"), on WiFi with the wired interface down. Ping is 33-134 ms with
32 ms jitter. Effective throughput measured ~1.6-1.7 MB/s, against ~124 MB/s for
the same payload on loopback — roughly 70x slower.

Everything that looked like a camera problem follows from that ceiling. Per-frame
cost was ~330 ms at full frame, nearly independent of exposure, and scaled with
ROI pixel count:

| ROI requested | pixels | bytes | median per frame | implied rate |
|---|---|---|---|---|
| 512x512 | 262k | 524 kB | 330 ms | 1.7 MB/s |
| 256x256 | 65k | 131 kB | ~100 ms | 1.6 MB/s |
| 128x128 | 16k | 33 kB | ~40 ms | 1.6 MB/s |
| 64x64 | 2k | 8 kB | ~25 ms | 1.6 MB/s |

A constant ~20 ms overhead plus a fixed 1.6-1.7 MB/s is the signature of a
bandwidth-limited link, not of camera readout. Enabling EM gain changed nothing
measurable (557 ms/frame, if anything worse), which is expected if the link, not
the sensor, sets the pace.

The same ceiling explains why the fix looked useless here: the defect costs
**CPU** (memcpy), and at 1.7 MB/s that cost hides entirely under network wait.
On a fast link the copying dominates instead, which is what the loopback
benchmark shows.

**To get a valid answer**, run `tools/bench_hamcam_rpc.py` *on PH-BEAST itself*
(loopback to its own device server) or from a machine wired to the lab LAN
(10.59.3.0/24). Do not run it across Tailscale.

### Two things this turned up that are NOT network artefacts

* **`set_roi` is off by one.** pylablib's `set_roi(hstart, hend, ...)` treats
  `hend` as *exclusive* (it uses `hend - hstart` as the size), but
  `imagemX2.py:60` passes `x0 + width - 1`. Every explicit ROI request is one
  pixel short and then truncated to the camera's granularity: asking for 512
  returns **496**. Not fixed here — analysis code may already be compensating.
* **`"sequence"` mode returns empty arrays.** The buffer window advances past
  absolute frame 0, so `read_multiple_images((0, n))` finds nothing. Use
  `"snap"`. Also, >~25 frames per call exceeds the server-side 20 s
  `wait_for_frame` timeout at 330 ms/frame.

## Fix applied

- **`pytweezer/servers/sipyco_patches.py`** (new): a linear replacement for
  `_socket_readline` — accumulates chunks in a list, joins once, and reads 64 KB
  per `recv()` instead of 4 KB. `apply_patches()` is idempotent.
- **`pytweezer/servers/__init__.py`**: calls `apply_patches()` on import, so
  every pytweezer process that touches `pytweezer.servers` (which includes
  `get_device`, every driver, both GUIs) gets the fix automatically.
- **`tests/test_sipyco_patches.py`** (new, 8 tests): correctness of the
  replacement reader (partial reads, line splitting across chunks, EOF
  handling) plus a regression guard that fails if the read time scaling goes
  quadratic again.
- Full suite: `poetry run pytest tests/ -q` → 126 passed.

**Caveat**: the patch only takes effect in processes that `import
pytweezer.servers` (directly or transitively). A notebook or script using a
bare `sipyco.pc_rpc.Client` without importing anything from `pytweezer`
first would not get it — in practice this doesn't happen in this codebase,
since `get_device()` (which does the import) is the only way camera clients
are built.

## To decide tomorrow

1. **Merge `sipyco-fix` to `main`?** Self-contained, test-covered, no API
   change — low risk. My recommendation, but flagging since this branch
   hasn't been reviewed yet.
2. **Upstream to m-labs/sipyco?** This is a real bug in the library itself,
   not something pytweezer-specific — every project using
   `sipyco.pc_rpc.Client` for anything beyond tiny replies has this problem.
   Worth a PR against m-labs/sipyco; happy to write it if wanted. Until/unless
   that lands (and this project bumps to a fixed version), keeping the local
   patch is the right call regardless.
3. ~~`pybase64` isn't installed~~ — **done**, added to `pyproject.toml` and
   confirmed picked up (`sipyco.pyon` uses it automatically once importable;
   `pyon.base64.__name__ == "pybase64"`). Isolated its effect on just the
   PYON encode/decode step for a 100-frame (52 MB raw / 70 MB base64) reply:

   | | encode | decode | total |
   |---|---|---|---|
   | stdlib `base64` | 207 ms | 165 ms | 372 ms |
   | `pybase64` | 170 ms | 100 ms | 270 ms |

   ~27% off the encode/decode step, ~100 ms saved on a 100-frame grab. Small
   next to the readline fix's ~200 s → 0.5 s, but free and stacks with it —
   full pipeline for 100 frames is now 475–603 ms patched (vs. 590–700 ms
   before `pybase64`), full suite (126 tests) still green.

4. One thing still open, **not** touched:
   - The sipyco server's request size limit is `4*1024*1024` bytes
     (`sipyco/tools.py:84`, `asyncio.start_server(..., limit=4*1024*1024)`).
     That's *inbound* to the server (RPC method arguments), not the frame
     replies discussed above, but it means an RPC call passing >~3 MB of raw
     argument data (e.g. a large mask or config array pushed over RPC rather
     than in-process) would fail outright rather than merely being slow.
     Not currently hit anywhere I found — the SLM mask goes in-process via the
     rearrangement coordinator — but worth knowing if that ever changes.

## Reproducing / re-checking

```bash
poetry run pytest tests/test_sipyco_patches.py -q   # correctness + regression guard
poetry run python tools/bench_sipyco_readline.py     # synthetic server, ~1-2 min
poetry run python tools/bench_hamcam_rpc.py          # real camera over the real link
```

`tools/bench_hamcam_rpc.py` drives a real camera on its **internal** trigger, so
no MotMaster/external trigger is needed, and prints the per-call curve for both
arms. It only needs the device's server to be running (`Rb HamCam` is served by
`Rb Rearrangement Rig` on PH-BEAST, port 7296); because the fix is entirely
client-side, that server does **not** need the fix or a restart, and the
benchmark can be run from any machine that can reach it. Point it elsewhere with
`--device "CaF HamCam"` or an explicit `--device host:port/target`.

Both are currently **uncommitted on branch `sipyco-fix`** except for the
files already in commit `5aef1f6`
(`pytweezer/servers/sipyco_patches.py`, `pytweezer/servers/__init__.py`,
`tests/test_sipyco_patches.py`). `tools/bench_sipyco_readline.py` and this
doc are new and not yet staged/committed.
