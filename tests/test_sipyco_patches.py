import time

from sipyco import pc_rpc

from pytweezer.servers import sipyco_patches


class FakeSocket:
    """Hands out a canned byte stream in fixed-size pieces, like a real ``recv``."""

    def __init__(self, payload, chunk=4096):
        self._payload = payload
        self._chunk = chunk
        self._pos = 0
        self.recv_calls = 0

    def recv(self, bufsize):
        self.recv_calls += 1
        size = min(bufsize, self._chunk)
        piece = self._payload[self._pos : self._pos + size]
        self._pos += len(piece)
        return piece


def test_patch_is_installed_on_import_and_reapplying_keeps_it():
    assert pc_rpc._socket_readline is sipyco_patches._socket_readline
    sipyco_patches.apply_patches()
    assert pc_rpc._socket_readline is sipyco_patches._socket_readline


def test_reassembles_a_long_line_with_large_recvs():
    payload = b"y" * 200_000 + b"\n"
    sock = FakeSocket(payload, chunk=sipyco_patches.RECV_BUFSIZE)
    assert sipyco_patches._socket_readline(sock) == payload.decode()
    # 200 kB at the sipyco default of 4 kB would need >49 recvs.
    assert sock.recv_calls <= 6


def test_read_time_scales_linearly_not_quadratically():
    """Guards the actual defect: the stock loop is O(n^2) in the reply size."""

    def elapsed(nbytes):
        payload = b"z" * nbytes + b"\n"
        best = float("inf")
        for _ in range(3):
            sock = FakeSocket(payload, chunk=4096)
            start = time.perf_counter()
            sipyco_patches._socket_readline(sock)
            best = min(best, time.perf_counter() - start)
        return best

    # Free one large block first: glibc then serves multi-MB buffers from the
    # heap rather than fresh mmaps, so the ratio below measures the read loop
    # and not page faults, whatever earlier tests left the allocator in.
    warm_up = bytearray(24_000_000)
    del warm_up
    small = elapsed(2_000_000)
    large = elapsed(8_000_000)
    # Linear would be ~4x. The stock quadratic loop lands near 16x or worse;
    # the ceiling is loose enough to absorb allocator noise on a busy machine.
    assert large < small * 9
