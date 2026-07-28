"""Standalone benchmark for the sipyco `_socket_readline` quadratic-read bug.

Spins up a real sipyco RPC server on loopback returning uint16 image frames,
then times a stock `sipyco.pc_rpc.Client` against one with
`pytweezer.servers.sipyco_patches` applied, each in a fresh subprocess so
interpreter warm-up can't leak between measurements.

Usage:
    poetry run python tools/bench_sipyco_readline.py
"""

import subprocess
import sys
import time

import numpy as np
from sipyco.pc_rpc import Client, simple_server_loop

PORT = 45951
H = W = 512


class _Target:
    def frames(self, n):
        return np.zeros((int(n), H, W), dtype=np.uint16)


def _run_server():
    simple_server_loop({"camera": _Target()}, "127.0.0.1", PORT, "bench")


def _run_client(which, n):
    if which == "patched":
        import pytweezer.servers  # noqa: F401 -- installs the patch on import
    c = Client("127.0.0.1", PORT, target_name="camera")
    t = time.perf_counter()
    a = c.frames(n)
    dt = time.perf_counter() - t
    c.close_rpc()
    assert a.shape == (n, H, W) and a.dtype == np.uint16
    print(f"{dt:.6f}")
    sys.stdout.flush()


def main():
    srv = subprocess.Popen([sys.executable, __file__, "server"])
    time.sleep(3)
    try:
        print(f"{'frames':>7} {'raw MB':>7} {'stock ms':>11} {'patched ms':>12} {'speedup':>9}")
        for n in (1, 10, 50, 100):
            out = {}
            for which in ("stock", "patched"):
                r = subprocess.run(
                    [sys.executable, __file__, "client", which, str(n)],
                    capture_output=True, text=True,
                )
                if r.returncode != 0:
                    print(r.stdout, r.stderr, file=sys.stderr)
                    raise SystemExit(1)
                out[which] = float(r.stdout.strip())
            print(
                f"{n:7d} {n * H * W * 2 / 1e6:7.1f} {out['stock'] * 1e3:11.1f} "
                f"{out['patched'] * 1e3:12.1f} {out['stock'] / out['patched']:8.1f}x",
                flush=True,
            )
    finally:
        srv.terminate()
        srv.wait()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        _run_server()
    elif len(sys.argv) > 1 and sys.argv[1] == "client":
        _run_client(sys.argv[2], int(sys.argv[3]))
    else:
        main()
