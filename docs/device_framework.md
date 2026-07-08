# pytweezer device control framework

How device RPC servers and clients are defined, launched, and connected to.
Complements [`gui_architecture.md`](gui_architecture.md), which covers the GUIs
that embed the **Devices** tab and the cross-PC **Device Status** feed described
here only in passing.

## The problem this solves

Every physical device (a MotMaster experiment sequencer, a camera) is driven by
a small [sipyco](https://github.com/m-labs/sipyco) RPC server process running on
whichever PC the hardware is attached to. Before this framework, each driver
module (`motmaster_server.py`, `imagemX2.py`, `bfly2.py`) duplicated its own
`argparse` parser, config-reading, backend construction, and
`simple_server_loop(...)` call. Adding a device meant copy-pasting that
boilerplate; the client side had the same duplication (`MotMasterClient`, one
per device type).

Now there is **one generic server launcher** and **one generic client factory**.
Adding a new *instance* of an existing device type is a config-only change;
adding a new *driver type* is one factory function.

## Component map

```
pytweezer/configuration/config.py
  CONFIG["Devices"] = { name: {"driver": ..., "host":..., "port":..., ...}, ... }

pytweezer/servers/device_server.py      (generic server launcher)
├── DRIVER_REGISTRY   driver-key -> factory(name, conf) -> DeviceServerSpec
├── _make_motmaster / _make_imagemx2 / _make_blackfly   (lazy backend imports)
├── resolve_device(name)   lenient config-key lookup
├── build_spec(name)       config -> DeviceServerSpec
├── run_device_server(name, host=None, port=None)   blocks, serves forever
└── main()                 CLI entry: pytweezer-device <name>

pytweezer/servers/device_client.py       (generic client factory)
├── get_device_config(name)
└── get_device(name, host=None, port=None, target_name=AutoTarget, timeout=None)
      -> sipyco.pc_rpc.Client (transparent RPC proxy)

pytweezer/servers/simulated_device.py    (generic simulated-backend mechanism)
├── public_methods(real_cls, exclude=())   introspects real_cls's public methods
└── simulate(real_cls, exclude=(), defaults=None)   class decorator: auto-stub
      any public method of real_cls a decorated class doesn't already define

bin/process_manager.py            DeviceManager (host-filtered start/stop tiles)
bin/process_tile_base.py          ProcessTile (subprocess: python <script> <name>)
pytweezer/servers/device_status.py DeviceStatusServer/-Client (cross-PC up/down feed)
```

## Server side: one launcher for every device

### Config shape

Each entry in `CONFIG["Devices"]` (`pytweezer/configuration/config.py`) now has a
`"driver"` key alongside the usual `host`/`port`/`active`/`script`:

```python
"Rb HamCam": {
    "active": True,
    "script": "../pytweezer/servers/device_server.py",
    "driver": "imagemx2",
    "host": SERVER_HOST,
    "port": get_next_port(),
    "simulate": SIMULATING,
    "stream_name": "rb_hamcam",
    "timeout": 5.0,
    "image_dir": "...",
},
```

`"script"` points at the **same file for every device** —
`pytweezer/servers/device_server.py` — so `ProcessTile` always runs
`python device_server.py <name>` regardless of device type. `"driver"` is what
actually selects the behavior.

### `DRIVER_REGISTRY`: driver key → factory

`device_server.py` maps a `"driver"` string to a factory function:

| `"driver"` | Factory | Backend built | Target name |
|---|---|---|---|
| `"motmaster"` | `_make_motmaster` | `MotMasterInterface` / `SimulatedMotMasterInterface` (if `simulate`) | `"motmaster"` |
| `"imagemx2"` | `_make_imagemx2` | `ImagEMX2Camera` / `SimulatedImagEMX2Camera` (if `simulate`) | `"camera"` |
| `"blackfly"` | `_make_blackfly` | `Blackfly` | `"camera"` |

Each factory returns a `DeviceServerSpec(target_name, target, description,
teardown=None)`. `run_device_server` does the generic part:

```python
spec = build_spec(name)                  # config -> DeviceServerSpec
simple_server_loop({spec.target_name: spec.target}, host=host, port=port,
                    description=spec.description)
# ... then, in finally: spec.teardown() if provided (e.g. MotMaster disconnect)
```

**Backend imports are lazy** (inside each factory, not at module top). This
matters because `_make_blackfly` imports `rotpy`, a hardware library that may
not be installed on every machine — importing `device_server.py` itself must
never fail just because one driver's hardware library is absent. Only launching
*that specific* device pulls in its library.

**Simulate-mode is respected per driver.** `_make_motmaster` only calls
`_ensure_motmaster_running()` (which launches the real `MOTMaster.exe`) in the
non-simulate branch — a simulated MotMaster server never touches real hardware.

### `simulate()`: one mechanism for every driver's simulated backend

`pytweezer/servers/simulated_device.py` provides a class decorator,
`simulate(real_cls, *, exclude=(), defaults=None)`, used to build both
`SimulatedMotMasterInterface` (`pytweezer/experiment/motmaster_server.py`) and
`SimulatedImagEMX2Camera` (`pytweezer/drivers/imagemX2.py`). Rather than
hand-copying every method of the real backend class into a parallel
`Simulated*` class — which drifts out of sync silently as the real class
changes — `simulate()` inspects `real_cls`'s public methods at class
*definition* time (never an instance, so it never touches hardware) and
injects a safe, logging, no-op stub for every one the decorated class doesn't
already define itself:

```python
@simulate(ImagEMX2Camera)
class SimulatedImagEMX2Camera:
    def __init__(self, image_dir=None, timeout=5.0, stream_name=None):
        ...
    def _generate_frame(self) -> np.ndarray:
        ...  # Gaussian-noise background + scattered atom blobs
    def acquire_n_frames(self, nframes, start_frame=0, autosave=False, broadcast=False):
        ...
    # set_ccd_mode, enable_em_gain, set_sensitivity, ... are all auto-stubbed
```

Only methods with genuinely interesting fake behavior need a hand-written
body; everything else stays interface-complete automatically, and stubs
return plain `None`/`dict`/`list` values (never a `unittest.mock.Mock`) so
they remain serializable by `sipyco.pyon` when a client calls them over RPC —
a bare `MagicMock`/`create_autospec` object is **not** PYON-encodable and
breaks every unconfigured RPC call, which is why this repo doesn't use one as
the literal RPC target. `defaults={method_name: value_or_zero_arg_callable}`
overrides the stub's return value for methods where `None` would be a poor
fake (e.g. `dict`/`list` for methods real callers expect to iterate over).

### Adding a new driver type

1. Write the backend class as normal (own module, own connect/acquire/etc.
   methods) — no RPC or config code in it. If it needs a simulated variant,
   write a small `Simulated<X>` class decorated with `@simulate(<X>)` next to
   it, hand-implementing only the methods with interesting fake behavior.
2. Add one `_make_<driver>(name, conf)` factory to `device_server.py` that
   builds the backend from `conf` and returns a `DeviceServerSpec`. Import the
   backend module lazily, inside the factory.
3. Register it: `DRIVER_REGISTRY["<driver>"] = _make_<driver>`.
4. Add device instances to `CONFIG["Devices"]` with
   `"script": "../pytweezer/servers/device_server.py"` and
   `"driver": "<driver>"`.

No new console script, no new argparse, no new `simple_server_loop` call.

### Adding a new instance of an existing driver type

Config-only: add an entry to `CONFIG["Devices"]` with an existing `"driver"`
value and that driver's expected keys (see the table above / factory source for
which config keys each factory reads).

### Launching a device server

Two equivalent paths, both ending up at `run_device_server(name)`:

- **CLI:** `pytweezer-device "Rb HamCam"` (console script in `pyproject.toml`,
  → `pytweezer.servers.device_server:main`). Also runnable as
  `python -m pytweezer.servers.device_server "Rb HamCam"` without `poetry
  install`.
- **Device manager GUI:** `DeviceManager` (`bin/process_manager.py`) builds a
  `ProcessTile` per host-matching, active device; the tile's start button runs
  `python <script> <name>` — same module, same `main()`.

`--host`/`--port` CLI flags override the config's values if given.

### Lenient device-name matching (`resolve_device`)

Device names in config often contain spaces (`"Rb HamCam"`), which are awkward
to type as a bare CLI argument. `resolve_device(name)`:

1. Tries an exact match against `CONFIG["Devices"]` keys first.
2. Falls back to a whitespace-stripped, lowercased comparison — so
   `RbHamCam`, `rb hamcam`, and `RB HAMCAM` all resolve to the config key
   `"Rb HamCam"`.
3. If nothing matches, raises `KeyError` listing every valid device name, so a
   typo fails fast with an actionable message instead of a confusing downstream
   error (e.g. "no port configured").

`run_device_server` and `build_spec` both go through `resolve_device`, so both
the CLI and the device manager benefit.

## Client side: one factory for every device

`pytweezer/servers/device_client.py` gives callers (experiment scripts,
notebooks, other servers) a way to get an RPC client without writing a client
class per device:

```python
from pytweezer.servers.device_client import get_device

cam = get_device("Rb HamCam")
cam.acquire()          # remote call, transparently proxied
cam.close_rpc()
```

`get_device(name, host=None, port=None, target_name=AutoTarget, timeout=None)`
looks up `name` in `CONFIG["Devices"]`, resolves `host`/`port` (overridable),
and returns a `sipyco.pc_rpc.Client`. It defaults `target_name` to
`sipyco.pc_rpc.AutoTarget`, which auto-selects the server's sole RPC target —
since every device server in this framework exposes exactly one target
(`"motmaster"` or `"camera"`), the caller never needs to know or hardcode that
name.

`get_device_config(name)` is available if you just want the raw config dict
(host/port/driver/etc.) without connecting.

Unknown device names raise `KeyError` listing the valid names (same pattern as
`resolve_device`, but this lookup is **exact-match only** — it does not apply
the lenient normalization, since callers here are code, not a human typing a
CLI arg).

### Relationship to `MotMasterClient`

`pytweezer/experiment/motmaster_client.py`'s `MotMasterClient` predates this
framework and still exists for its back-compat method aliases
(`set_script`/`start_experiment`/`shutdown_server` → the real
`set_motmaster_experiment`/`start_motmaster_experiment`/`terminate`). New code
should prefer `get_device(...)` directly; `MotMasterClient` has not been
migrated onto it.

## How this fits the rest of the system

- **`DeviceManager`** (`bin/process_manager.py`) still does its own host
  filtering (`check_host`) and start/stop tile management — it only changed in
  that every device's tile now launches the same `device_server.py` script.
- **`DeviceStatusServer`** (`pytweezer/servers/device_status.py`) independently
  TCP-probes `host:port` for every `CONFIG["Devices"]` entry to publish a
  cross-PC up/down feed — it does not use `get_device`/RPC calls, just raw
  reachability (see `gui_architecture.md` for the full Device Status writeup).
- **Both GUIs' Applets/experiment code** are the intended callers of
  `get_device` going forward, instead of hand-rolled `sipyco.pc_rpc.Client(...)`
  calls with hardcoded host/port.

## File index

| File | Role |
|------|------|
| `pytweezer/servers/device_server.py` | Generic server launcher: `DRIVER_REGISTRY`, `build_spec`, `resolve_device`, `run_device_server`, `main`. |
| `pytweezer/servers/device_client.py` | Generic client factory: `get_device`, `get_device_config`. |
| `pytweezer/servers/simulated_device.py` | Generic simulated-backend mechanism: `simulate` class decorator, `public_methods`. |
| `pytweezer/experiment/motmaster_server.py` | MotMaster backend classes (`MotMasterInterface`, `SimulatedMotMasterInterface`) + legacy standalone `main()`. |
| `pytweezer/drivers/imagemX2.py` | ImagEM X2 backend classes (`ImagEMX2Camera`, `SimulatedImagEMX2Camera`) + legacy standalone `main()`. |
| `pytweezer/drivers/bfly2.py` | Blackfly backend class + legacy standalone `main()`. |
| `pytweezer/experiment/motmaster_client.py` | `MotMasterClient` — pre-framework client with back-compat aliases. |
| `pytweezer/configuration/config.py` | `CONFIG["Devices"]` — `driver`/`host`/`port`/`active`/... per device. |
| `bin/process_manager.py` | `DeviceManager` — host-filtered start/stop tiles (unchanged besides the shared script path). |
| `pyproject.toml` | `pytweezer-device = "pytweezer.servers.device_server:main"` console script. |

**Note:** each driver module's original standalone `main()`/`run_server()` is
still present (useful for ad hoc debugging directly against that module) but is
no longer on the launch path — both the CLI and the device manager go through
`device_server.py`.
