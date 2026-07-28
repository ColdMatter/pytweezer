# HamCam RPC slowness — investigation notes (2026-07-28)

Status: root-caused and fixed on branch `sipyco-fix` (commit `5aef1f6`),
**not merged, not pushed**. Written up here for review.

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
| 1 | 0.5 MB | 20–29 ms | 5–8 ms | ~4× |
| 10 | 5.2 MB | 1.7–1.9 s | 47–59 ms | ~35× |
| 50 | 26.2 MB | 48.7–54.8 s | 306–371 ms | ~150× |
| 100 | 52.4 MB | **197–207 s** | **0.59–0.70 s** | ~300× |

This matches the real `hybrid_exp_test.ipynb` / `28Jul26Exp.ipynb` /
`rearrangement_server_test.ipynb` usage pattern directly — e.g.
`imgs_ham = hamcam.acquire_n_frames(n_iterations)` and
`local_cam.acquire_n_frames(n_iterations * 2)` — any loop that grabs more
than a handful of frames per RPC call pays this.

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
3. **Two adjacent things noticed, not touched, possibly worth separate
   tickets:**
   - `pybase64` isn't installed (`pyproject.toml`/`poetry.lock` only pull
     stdlib `base64` as sipyco's fallback). Now that the O(n²) term is gone,
     base64 encode/decode is the next-largest cost (~30 ms of the 0.59 s for
     a 100-frame grab). Installing `pybase64` would shave more off, if this
     matters (e.g. for the rearrangement loop's tight timing).
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
poetry run python tools/bench_sipyco_readline.py     # before/after timing, ~1-2 min
```

Both are currently **uncommitted on branch `sipyco-fix`** except for the
files already in commit `5aef1f6`
(`pytweezer/servers/sipyco_patches.py`, `pytweezer/servers/__init__.py`,
`tests/test_sipyco_patches.py`). `tools/bench_sipyco_readline.py` and this
doc are new and not yet staged/committed.
