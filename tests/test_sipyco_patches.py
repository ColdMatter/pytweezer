"""Tests for the sipyco runtime patches in :mod:`pytweezer.servers.sipyco_patches`."""

import time

import pytest
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


def test_patch_is_installed_on_import():
    assert pc_rpc._socket_readline is sipyco_patches._socket_readline


def test_apply_patches_is_idempotent():
    sipyco_patches.apply_patches()
    sipyco_patches.apply_patches()
    assert pc_rpc._socket_readline is sipyco_patches._socket_readline


def test_reads_a_whole_line():
    sock = FakeSocket(b"hello world\n")
    assert sipyco_patches._socket_readline(sock) == "hello world\n"


def test_reassembles_a_line_split_across_many_chunks():
    payload = b"x" * 100_000 + b"\n"
    sock = FakeSocket(payload, chunk=4096)
    assert sipyco_patches._socket_readline(sock) == payload.decode()


def test_stops_at_the_first_newline_without_consuming_the_rest():
    sock = FakeSocket(b"first\nsecond\n", chunk=4)
    line = sipyco_patches._socket_readline(sock)
    assert line.startswith("first\n")


def test_raises_eof_when_the_peer_closes_mid_line():
    sock = FakeSocket(b"no newline here")
    with pytest.raises(EOFError):
        sipyco_patches._socket_readline(sock)


def test_uses_a_large_default_recv_size():
    payload = b"y" * 200_000 + b"\n"
    sock = FakeSocket(payload, chunk=sipyco_patches.RECV_BUFSIZE)
    sipyco_patches._socket_readline(sock)
    # 200 kB at the sipyco default of 4 kB would need >49 recvs.
    assert sock.recv_calls <= 6


def test_read_time_scales_linearly_not_quadratically():
    """Guards the actual defect: the stock loop is O(n^2) in the reply size.

    Quadrupling the payload must not multiply the time by anything near 16.
    """

    def elapsed(nbytes):
        payload = b"z" * nbytes + b"\n"
        best = float("inf")
        for _ in range(3):
            sock = FakeSocket(payload, chunk=4096)
            start = time.perf_counter()
            sipyco_patches._socket_readline(sock)
            best = min(best, time.perf_counter() - start)
        return best

    small = elapsed(2_000_000)
    large = elapsed(8_000_000)
    # Linear would be ~4x. The stock quadratic loop lands near 16x or worse;
    # the ceiling is loose enough to absorb allocator noise on a busy machine.
    assert large < small * 9
