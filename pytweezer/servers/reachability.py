"""Tiny TCP reachability probe shared by the GUI status panels and the
device-status server.

A successful ``socket.create_connection`` means the target is bound and
accepting connections, which for our ZMQ hubs and sipyco device servers is a
sufficient liveness signal.
"""

import socket


def is_reachable(host, port, timeout=0.3):
    """Return ``True`` if a TCP connection to ``(host, port)`` succeeds within
    ``timeout`` seconds, ``False`` otherwise.

    The short timeout keeps callers from blocking on a down or unreachable host.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
