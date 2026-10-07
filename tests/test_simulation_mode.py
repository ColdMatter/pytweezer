"""pytweezer-server simulates everywhere except the server PC.

CONFIG is computed at import, so each case imports it in a fresh interpreter.
"""

import json
import os
import subprocess
import sys
import types

from bin import launch

PROBE = """
import json, socket, sys
if len(sys.argv) > 1:
    socket.gethostname = lambda: sys.argv[1]
from pytweezer.configuration import config
c = config.CONFIG
print(json.dumps({
    "simulating": config.SIMULATING,
    "forced": config.SIMULATION_FORCED,
    "server_host": config.SERVER_HOST,
    "manager_simulate": c["Servers"]["Experiment Manager"]["simulate"],
    "manager_host": c["Servers"]["Experiment Manager"]["host"],
    "device_simulate": c["Devices"]["Rb MotMaster"]["simulate"],
}))
"""


def probe(role=None, hostname=None):
    env = {k: v for k, v in os.environ.items() if k != "PYTWEEZER_ROLE"}
    if role is not None:
        env["PYTWEEZER_ROLE"] = role
    args = [sys.executable, "-c", PROBE] + ([hostname] if hostname else [])
    result = subprocess.run(args, env=env, capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_server_session_off_the_server_pc_simulates_everything():
    state = probe(role="server", hostname="someones-laptop")
    assert state == {
        "simulating": True,
        "forced": True,
        "server_host": "127.0.0.1",
        "manager_simulate": True,
        "manager_host": "127.0.0.1",
        "device_simulate": True,
    }


def test_server_session_on_the_server_pc_is_real():
    state = probe(role="server", hostname="ph-beast")
    assert not state["simulating"] and not state["forced"]
    assert state["server_host"] == "10.59.3.1"


def test_other_processes_follow_the_flag_only():
    # e.g. pytweezer-client or a device server on a lab PC
    state = probe(hostname="IC-CZC4287H3W")
    assert not state["simulating"] and not state["forced"]


def test_launcher_marks_the_session_before_the_gui_imports(monkeypatch):
    monkeypatch.delenv("PYTWEEZER_ROLE", raising=False)
    seen = {}
    fake_gui = types.ModuleType("bin.gui")
    fake_gui.server_main = lambda: seen.update(role=os.environ.get("PYTWEEZER_ROLE"))
    fake_gui.client_main = lambda: seen.update(client=os.environ.get("PYTWEEZER_ROLE"))
    monkeypatch.setitem(sys.modules, "bin.gui", fake_gui)

    launch.client_main()
    assert seen == {"client": None}
    launch.server_main()
    assert seen["role"] == "server"


def test_launcher_imports_no_pytweezer_code():
    code = "import sys, bin.launch; print(any(m.startswith('pytweezer') for m in sys.modules))"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
