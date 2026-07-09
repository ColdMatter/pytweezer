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

4. **Resolved: `Go()` blocks for the full sequence.** `MotMasterInterface.start_motmaster_experiment()`
   → `self.motmaster.Go()` (`pytweezer/experiment/motmaster_interface.py:190`) is a
   .NET remoting call (`Activator.GetObject`) — it's a synchronous proxy call that
   returns only once the remote `Go()` returns, and every scan helper in that file
   calls it in a loop with `time.sleep(self.interval)` right after, which only makes
   sense if `Go()` already blocked for the shot. So calling two MotMasters through
   plain blocking `get_device()` clients **sequentially** runs them back-to-back, not
   in parallel — each `start_motmaster_experiment()` call doesn't return until that
   MotMaster's sequence finishes. Each MotMaster is already its own device server
   process, so the two hardware sets don't contend with each other; the only thing
   serializing them is the client script awaiting one RPC reply before issuing the
   next. Bump `cam`'s `timeout` above the 5s default if reading after a `Go()` that
   runs long.

5. **Done: `get_device_async()` added** to `pytweezer/servers/device_client.py`,
   using `sipyco.pc_rpc.AsyncioClient` (~15 lines, no server changes). To run two
   MotMasters in parallel, connect both and drive them with `asyncio.gather`:

   ```python
   import asyncio
   from pytweezer.servers.device_client import get_device_async

   async def main():
       mm1 = await get_device_async("Rb MotMaster Server")
       mm2 = await get_device_async("CaF MotMaster Server")
       try:
           await asyncio.gather(
               mm1.start_motmaster_experiment(),
               mm2.start_motmaster_experiment(),
           )
       finally:
           await mm1.close_rpc()
           await mm2.close_rpc()

   asyncio.run(main())
   ```

   This works because `AsyncioClient` awaits the full round trip per call (per
   note 1) but `asyncio.gather` issues both calls before awaiting either, so the
   two `Go()` calls fire close together instead of one waiting for the other to
   finish. Caveat: GUI is PyQt5 with no `qasync`, so calling from GUI code needs a
   worker thread; fine from scripts/notebooks. Tests: `tests/test_servers.py`
   (`test_get_device_async_*`).

6. **If servers need to stay responsive during a blocking call** (abort a hung
   acquisition, keep status panels alive): make slow methods `async def` +
   `asyncio.to_thread(...)`, and pass `allow_parallel=True` through
   `run_device_server()` → `simple_server_loop()` (not currently exposed). Only
   worth doing if there's a concrete need.

7. **Skip `sipyco.fire_and_forget.FFProxy`** — it drops the return value and
   downgrades exceptions to warnings; you'd lose the acquired image.

## Next steps (pick up here)

- Only pursue server-side `to_thread`/`allow_parallel` changes if responsiveness
  during blocking calls becomes a real requirement (e.g. needing to abort one
  MotMaster while the other is still running its sequence).
</content>
