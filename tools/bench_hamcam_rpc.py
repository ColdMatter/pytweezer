"""Time real HamCam frame grabs over RPC, with and without the sipyco readline fix.

Companion to ``bench_sipyco_readline.py``, which measures the same defect against
a synthetic server on loopback. This one talks to a real camera through a real
device server, so it also captures the camera's own acquisition cost and the
network path between this machine and the server.

The camera is put on its **internal** trigger, so it free-runs and no external
trigger source (MotMaster) is needed.

The fix is entirely client-side -- it replaces the reply reader used by
``sipyco.pc_rpc.Client`` -- so the device server does not need to be restarted,
or even to have the fix, for these numbers to be meaningful. Each measurement
runs in a fresh subprocess so interpreter warm-up cannot leak between arms.

**Run this on the lab LAN.** Over a VPN or a relayed Tailscale link the reply is
bandwidth-limited, which hides the CPU cost the fix removes and makes the two
arms look identical. Check the reported MB/s: a flat few MB/s across all frame
counts means you are measuring the link, not the camera or the fix.

Usage (from the machine you normally drive the experiment from)::

    poetry run python tools/bench_hamcam_rpc.py
    poetry run python tools/bench_hamcam_rpc.py --device "CaF HamCam"
    poetry run python tools/bench_hamcam_rpc.py --frames 1,5,10 --exposure 0.005

Requires the device's server to already be running (start it from the GUI's
Devices tab, or ``poetry run pytweezer-device "Rb Rearrangement Rig"``).
"""

import argparse
import subprocess
import sys
import time

DEFAULT_DEVICE = "Rb HamCam"
DEFAULT_FRAMES = (1, 5, 10, 25, 50)


def _stock_socket_readline(socket, bufsize=4096):
    """Verbatim copy of the stock ``sipyco.pc_rpc._socket_readline``.

    Reinstated for the "stock" arm, since importing anything from
    ``pytweezer.servers`` installs the fix automatically.
    """
    buf = socket.recv(bufsize).decode()
    offset = 0
    while buf.find("\n", offset) == -1:
        more = socket.recv(bufsize)
        if not more:
            raise EOFError("Connection closed before a full line was received.")
        buf += more.decode()
        offset += len(more)
    return buf


def _connect(device):
    """Return a client for ``device``, which is either a config device name or an
    explicit ``host:port/target`` address (handy for pointing at a one-off server)."""
    if ":" in device:
        hostport, _, target = device.partition("/")
        host, _, port = hostport.partition(":")
        from sipyco.pc_rpc import Client

        return Client(host, int(port), target_name=target or None)

    from pytweezer.servers.device_client import get_device

    return get_device(device)


def _set_ccd_mode(cam, em):
    """Select the ImagEM's readout port: EM register (``em=True``) or conventional.

    This choice dominates the measurement, so it is not incidental setup. The
    two ports run at wildly different rates -- the ImagEM X2 reports
    ``internal_frame_rate`` of 70.4 fps on the EM register against 2.37 fps
    conventional, i.e. 14.2 ms vs 422 ms per frame at 512x512. Benchmarking the
    conventional port buries the RPC cost under half a second of acquisition per
    frame and makes the readline fix look irrelevant.

    Changing ``ccd_mode`` needs the camera idle, and a merely stopped
    acquisition still holds its DCAM buffers -- hence the close/reopen fallback.

    Raises if the mode could not be set: acquiring with EM gain in an unknown
    state is not something to do silently. Cameras without EM gain (ThorCam,
    ...) will raise here too -- run them with ``--no-setup``.
    """
    try:
        cam.stop_acquisition()
    except Exception:
        pass
    try:
        cam.enable_em_gain(em)
    except Exception:
        cam.relinquish_camera()
        time.sleep(1.0)
        cam.reacquire_camera()
        cam.enable_em_gain(em)


def _run_client(which, device, nframes, exposure, setup, repeats):
    import statistics

    import pytweezer.servers  # noqa: F401 -- installs the fix on import
    from sipyco import pc_rpc

    if which == "stock":
        pc_rpc._socket_readline = _stock_socket_readline

    cam = _connect(device)

    if setup:
        _disable_em_gain(cam)

    # Baseline round-trip on a tiny reply, to separate link latency from payload
    # transfer. Unaffected by the defect, so it should match across both arms.
    t = time.perf_counter()
    cam.is_connected()
    rtt = time.perf_counter() - t

    # Every call is timed and reported, with no warm-up discarded. The stock
    # reader's penalty is worst on the first grabs and shrinks as the allocator
    # settles into reusing one arena for the reply buffer, so a single averaged
    # number is misleading in either direction -- the shape of the curve is the
    # result. See the header of docs/hamcam_rpc_slowness_investigation.md.
    times = []
    for _ in range(repeats):
        if setup:
            # Re-arm every pass: "snap" completes after nframes, and a finished
            # acquisition has no new frames for wait_for_frame() to return.
            cam.set_trigger_source("int")
            cam.set_exposure_time(exposure)
            cam.setup_acquisition("snap", nframes)
            cam.start_acquisition()
        t = time.perf_counter()
        images = cam.acquire_n_frames(nframes)
        times.append(time.perf_counter() - t)
        if setup:
            cam.stop_acquisition()

    print(
        f"{times[0]:.6f} {statistics.median(times):.6f} {rtt:.6f} {images.nbytes} "
        + "x".join(str(d) for d in images.shape) + " "
        + ",".join(f"{x:.6f}" for x in times)
    )
    sys.stdout.flush()


def _measure(device, nframes, exposure, setup, repeats, which):
    cmd = [
        sys.executable, __file__, "client", which, device, str(nframes),
        str(exposure), "1" if setup else "0", str(repeats),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr, file=sys.stderr)
        raise SystemExit(f"{which} arm failed at nframes={nframes}")
    first, med, rtt, nbytes, shape, series = r.stdout.strip().split(" ")
    return (
        float(first), float(med), float(rtt), int(nbytes), shape,
        [float(x) for x in series.split(",")],
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", default=DEFAULT_DEVICE, help="device name to benchmark")
    p.add_argument(
        "--frames",
        default=",".join(str(n) for n in DEFAULT_FRAMES),
        help="comma-separated frame counts to sweep",
    )
    p.add_argument("--exposure", type=float, default=0.001, help="exposure time in seconds")
    p.add_argument(
        "--no-setup",
        action="store_true",
        help="skip trigger/exposure/acquisition setup (camera is already armed elsewhere)",
    )
    p.add_argument(
        "--stock-max-frames",
        type=int,
        default=50,
        help="skip the stock arm above this frame count (it is quadratic and slow)",
    )
    p.add_argument("--repeats", type=int, default=3, help="timed grabs per arm (median reported)")
    args = p.parse_args(argv)

    frames = [int(x) for x in args.frames.split(",") if x.strip()]
    setup = not args.no_setup

    print(
        f"device: {args.device}   exposure: {args.exposure}s   "
        f"setup: {setup}   repeats: {args.repeats}"
    )
    print(
        f"{'frames':>7} {'MB':>7} {'rtt ms':>8} | "
        f"{'stock 1st':>10} {'stock med':>10} | {'patch 1st':>10} {'patch med':>10} | "
        f"{'1st':>6} {'med':>6} | {'MB/s':>7}"
    )

    for n in frames:
        pf, pm, prtt, nbytes, shape, pser = _measure(
            args.device, n, args.exposure, setup, args.repeats, "patched"
        )
        if n <= args.stock_max_frames:
            sf, sm, _, _, _, sser = _measure(
                args.device, n, args.exposure, setup, args.repeats, "stock"
            )
            cols = (
                f"{sf * 1e3:10.1f} {sm * 1e3:10.1f} | {pf * 1e3:10.1f} {pm * 1e3:10.1f} | "
                f"{sf / pf:5.1f}x {sm / pm:5.1f}x"
            )
        else:
            sser = None
            cols = (
                f"{'skipped':>10} {'skipped':>10} | {pf * 1e3:10.1f} {pm * 1e3:10.1f} | "
                f"{'-':>6} {'-':>6}"
            )
        # Throughput of the fast arm. If this is flat across frame counts and far
        # below the link's rated speed, the run is bandwidth-limited and says
        # nothing about the readline fix -- check you are not on a VPN/relay.
        mbps = nbytes / 1e6 / pm
        print(
            f"{n:7d} {nbytes / 1e6:7.1f} {prtt * 1e3:8.1f} | {cols} | {mbps:7.1f}   {shape}",
            flush=True,
        )
        if sser is not None:
            print(f"{'':>26}   stock per call: " + " ".join(f"{x * 1e3:8.1f}" for x in sser))
        print(f"{'':>26} patched per call: " + " ".join(f"{x * 1e3:8.1f}" for x in pser), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        _, _, which, device, nframes, exposure, setup, repeats = sys.argv
        _run_client(
            which, device, int(nframes), float(exposure), setup == "1", int(repeats)
        )
    else:
        main()
