# Async/fire-and-forget device communication

**Goal:** Issue commands to devices (camera, MotMaster) without waiting for a reply —
e.g., arm the camera, tell MotMaster to run a script that triggers the camera,
without blocking in sequence.

## Key findings

1. **`sipyco.pc_rpc.AsyncioClient` is not fire-and-forget.** It still awaits every
   reply (holds a lock for the full round trip) — the win is running RPCs to
   *different servers* concurrently, not skipping the reply.

2. **`sipyco.pc_rpc.Server` is single-threaded asyncio.** Blocking driver methods
   (plain `def`, not `async def`) freeze the whole server while executing — no
   other calls (status probes, aborts) get through during that time.

3. **You likely don't need async at all.** The camera driver
   (`pytweezer/drivers/camera_base.py`) already splits arm vs. read:
   - `start_acquisition()` returns immediately (camera armed, waiting on ext trigger)
   - `_read_frames()` (inside `acquire_single_frame`/`acquire_n_frames`) is what
     blocks, in `wait_for_frame`
   - Frames buffer in hardware, so you can arm, trigger via MotMaster, then read
     afterward — plain blocking `get_device()` clients work fine:
     ```python
     cam.start_acquisition()
     mm.start_motmaster_experiment()
     image = cam.acquire_single_frame(broadcast=True)  # blocks until frame ready
     ```

4. **Open question (unresolved):** Does `MotMasterInterface.start_motmaster_experiment()`
   → `self.motmaster.Go()` (`pytweezer/experiment/motmaster_interface.py:190`) return
   immediately once the sequence starts, or block until it finishes? This determines
   whether the sequential recipe above is truly non-blocking or just "blocks in a
   different place." Either way it should work, but if `Go()` blocks for the full
   sequence, bump `cam`'s `timeout` above the 5s default since the read starts late.

5. **If concurrent in-flight calls are needed** (e.g., overlapping long scans across
   multiple devices): add a `get_device_async()` alongside `get_device()` in
   `pytweezer/servers/device_client.py` using `AsyncioClient` — straightforward,
   ~15 lines, no server changes needed. Caveat: GUI is PyQt5 with no `qasync`, so
   calling from GUI code needs a worker thread; fine from scripts/notebooks.

6. **If servers need to stay responsive during a blocking call** (abort a hung
   acquisition, keep status panels alive): make slow methods `async def` +
   `asyncio.to_thread(...)`, and pass `allow_parallel=True` through
   `run_device_server()` → `simple_server_loop()` (not currently exposed). Only
   worth doing if there's a concrete need.

7. **Skip `sipyco.fire_and_forget.FFProxy`** — it drops the return value and
   downgrades exceptions to warnings; you'd lose the acquired image.

## Next steps (pick up here)

- Check what `motmaster_interface.Go()` actually does (blocks vs returns immediately)
- If needed, implement `get_device_async()` in `device_client.py`
- Only pursue server-side `to_thread`/`allow_parallel` changes if responsiveness
  during blocking calls becomes a real requirement
</content>
