"""The pub/sub stream clients, through an in-process stand-in for the hub."""

import threading
import time

import numpy as np
import pytest
import zmq

from pytweezer.servers import clients


@pytest.fixture
def hub(monkeypatch):
    """An XSUB/XPUB proxy like ``xsub_xpub.py``, serving every hub name."""
    context = zmq.Context()
    xsub, xpub = context.socket(zmq.XSUB), context.socket(zmq.XPUB)
    sub_port = xsub.bind_to_random_port("tcp://127.0.0.1")
    pub_port = xpub.bind_to_random_port("tcp://127.0.0.1")

    def run():
        try:
            zmq.proxy(xsub, xpub)
        except zmq.ContextTerminated:
            pass
        finally:
            xsub.close(linger=0)
            xpub.close(linger=0)

    threading.Thread(target=run, daemon=True).start()
    conf = {"host": "127.0.0.1", "pub_port": pub_port, "sub_port": sub_port}
    servers = {"Imagehub": conf, "Datahub": conf}
    monkeypatch.setattr(clients, "get_config", lambda: {"Servers": servers})
    yield
    context.term()


def _send_until_received(sender, receiver, array):
    # A subscription takes a moment to reach the publisher, so resend until one lands.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        sender.send({}, array)
        message = receiver.recv()
        if message is not None:
            return message
    raise TimeoutError("nothing received")


@pytest.mark.parametrize("client_class", [clients.ImageClient, clients.DataClient])
@pytest.mark.parametrize(
    "array",
    [
        np.arange(6, dtype=np.uint16).reshape(2, 3).T,
        np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2],
    ],
    ids=["fortran-order", "strided"],
)
def test_an_array_arrives_as_sent(hub, client_class, array):
    sender = client_class("cam")
    receiver = client_class("viewer", recvtimeout=100, subscribe="cam")
    try:
        _channel, _header, received = _send_until_received(sender, receiver, array)
    finally:
        for client in (sender, receiver):
            client.pub_socket.close(linger=0)
            client.sub_socket.close(linger=0)
    np.testing.assert_array_equal(received, array)
