"""Runtime patches applied to the vendored ``sipyco`` RPC library.

Imported for its side effects by :mod:`pytweezer.servers`, so every pytweezer
process that talks to a device server gets them. :func:`apply_patches` is
idempotent and safe to call again.

Reply reading
-------------
``sipyco.pc_rpc.Client`` reads each reply with ``_socket_readline``, which
accumulates 4 kB chunks into a ``str`` with ``buf += chunk`` while testing
``buf.find("\\n", offset)`` in the ``while`` condition. That shape stops CPython
from appending in place, so every chunk copies the whole reply built so far and
the read costs O(n^2) in the reply size. Camera frames are the payloads big
enough for it to hurt: a 100-frame 512x512 uint16 grab (52 MB) takes ~200 s to
read on loopback, against ~0.7 s here. The replacement accumulates into a list
and joins once, and reads 64 kB per ``recv`` instead of 4 kB.

Only the synchronous ``Client`` is affected. ``AsyncioClient`` already reads
through ``asyncio``'s C-implemented ``readline`` with a 100 MB line limit.
"""

from sipyco import pc_rpc

#: ``recv`` size for reply reading. Large replies are camera frames of a few MB
#: to tens of MB, so the 4 kB sipyco default costs thousands of extra syscalls.
RECV_BUFSIZE = 64 * 1024

_sipyco_patched_applied = False


def _socket_readline(sock, bufsize=RECV_BUFSIZE):
    """Read one newline-terminated line from ``sock``, in time linear in its length.

    Signature-compatible with ``sipyco.pc_rpc._socket_readline``, including
    raising ``EOFError`` if the peer closes mid-line.
    """
    chunks = []
    while True:
        more = sock.recv(bufsize)
        if not more:
            raise EOFError("Connection closed before a full line was received.")
        chunks.append(more)
        if b"\n" in more:
            break
    return b"".join(chunks).decode()


def apply_patches():
    """Install the patches into ``sipyco``. Idempotent."""
    global _sipyco_patched_applied
    if _sipyco_patched_applied:
        return
    pc_rpc._socket_readline = _socket_readline
    _sipyco_patched_applied = True
