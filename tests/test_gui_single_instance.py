import os
import socket
import threading

import pytest

from bin.single_instance import GuiAlreadyRunning, claim_gui
from tests.helpers import until


@pytest.fixture
def port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def claim_from_another_launch(port):
    """Claim as a second launch does; its blocking waits need the first lock's loop running."""
    refusals = []

    def claim():
        try:
            claim_gui("client", port=port)
        except GuiAlreadyRunning as refusal:
            refusals.append(refusal)

    thread = threading.Thread(target=claim)
    thread.start()
    return thread, refusals


def test_a_second_gui_on_the_pc_is_refused_and_raises_the_running_one(qapp, port):
    running = claim_gui("server", port=port)
    raised = []
    running.raise_requested.connect(lambda: raised.append(True))

    thread, refusals = claim_from_another_launch(port)
    assert until(lambda: not thread.is_alive() and raised, qapp=qapp)

    (refusal,) = refusals
    assert refusal.owner["role"] == "server"
    assert refusal.owner["pid"] == os.getpid()
    assert "server GUI is already running on this PC" in str(refusal)
    running.close()


def test_a_closed_gui_frees_the_pc_for_the_next_launch(qapp, port):
    claim_gui("server", port=port).close()
    claim_gui("client", port=port).close()


def test_a_port_held_by_another_program_refuses_without_guessing(qapp, port):
    with socket.socket() as other:
        other.bind(("127.0.0.1", port))
        other.listen()
        with pytest.raises(GuiAlreadyRunning) as refusal:
            claim_gui("server", port=port)
    assert refusal.value.owner is None
    assert f"port {port}" in str(refusal.value)
