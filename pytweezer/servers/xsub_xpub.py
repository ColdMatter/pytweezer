from typing import Any

import zmq
import sys
import os
import signal
import time

sys.path.append('../../')
sys.path.append('../')
from pytweezer import *
import json
import argparse
from pytweezer.servers.configreader import ConfigReader
import threading
from zmq.utils.monitor import recv_monitor_message
from pytweezer.analysis.print_messages import print_error

EVENT_MAP = {}
# print("Event names:")
for name in dir(zmq):
    if name.startswith('EVENT_'):
        value = getattr(zmq, name)
        # print("%21s : %4i" % (name, value))
        EVENT_MAP[value] = name


def _terminate_stale_instances(instance_name: str, grace_s: float = 1.5) -> None:
    if os.name != "posix":
        return

    this_pid = os.getpid()
    proc_dir = "/proc"
    stale_pids = []

    try:
        for pid_str in os.listdir(proc_dir):
            if not pid_str.isdigit():
                continue
            pid = int(pid_str)
            if pid == this_pid:
                continue

            cmdline_path = os.path.join(proc_dir, pid_str, "cmdline")
            try:
                with open(cmdline_path, "rb") as f:
                    cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
            except Exception:
                continue

            if "xsub_xpub.py" in cmdline and instance_name in cmdline:
                stale_pids.append(pid)
    except Exception as error:
        print_error(f"Could not scan for stale instances: {error}", "warning")
        return

    if not stale_pids:
        return

    print_error(
        f"Found stale {instance_name} instance(s) {stale_pids}; terminating before bind.",
        "warning",
    )

    for pid in stale_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception as error:
            print_error(f"Failed SIGTERM to pid {pid}: {error}", "warning")

    deadline = time.time() + grace_s
    while time.time() < deadline:
        remaining = []
        for pid in stale_pids:
            try:
                os.kill(pid, 0)
                remaining.append(pid)
            except ProcessLookupError:
                continue
            except Exception:
                continue
        if not remaining:
            return
        time.sleep(0.1)

    for pid in stale_pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception as error:
            print_error(f"Failed SIGKILL to pid {pid}: {error}", "warning")


def run_server(name, pubbinding_override=None, subbinding_override=None, kill_stale=True):
    conf = ConfigReader.getConfiguration()
    name = name.split('/')[-1]
    c = conf['Servers'][name]
    host = c['host']
    pub_port = c['pub_port']
    sub_port = c['sub_port']
    pubbinding = pubbinding_override or f"tcp://{host}:{pub_port}"
    subbinding = subbinding_override or f"tcp://{host}:{sub_port}"
    #print(name + " starting on")
    #print('XSUB: ', subbinding)
    #print('XPUB: ', pubbinding)

    if kill_stale:
        _terminate_stale_instances(name)

    context = zmq.Context()
    sock_sub = context.socket(zmq.XSUB)
    sock_pub = context.socket(zmq.XPUB)
    sock_sub.bind(subbinding)
    sock_pub.bind(pubbinding)

    pub_mon = sock_pub.get_monitor_socket()
    pub_mon_thread = threading.Thread(target=event_monitor, args=(pub_mon, name, 'PUB'))
    pub_mon_thread.start()
    sub_mon = sock_sub.get_monitor_socket()
    sub_mon_thread = threading.Thread(target=event_monitor, args=(sub_mon, name, 'SUB'))
    sub_mon_thread.start()

    # run server forever
    zmq.proxy(sock_sub, sock_pub)

    # clean up(never reached)
    print("server {} cleanup was reached, the comment on line 30 lied".format(name))
    sock_sub.close()
    sock_pub.close()
    context.term()


def event_monitor(monitor: zmq.Socket, name, pub_sub) -> None:
    while monitor.poll():
        evt: dict[str, Any] = {}
        mon_evt = recv_monitor_message(monitor)
        evt.update(mon_evt)
        evt['description'] = EVENT_MAP[evt['event']]
        #if evt['event'] == zmq.EVENT_DISCONNECTED:
        #    print_error(name + ' ' + pub_sub + ' ' + f"Event: {evt}")
        #else:
        #    print(name, pub_sub, f"Event: {evt}")
        if evt['event'] == zmq.EVENT_MONITOR_STOPPED:
            break
    monitor.close()
    print()
    print("event monitor thread done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs='?', default=None, help='name of this program instance')
    parser.add_argument('--pub', default=None, help='Optional XPUB bind address override')
    parser.add_argument('--sub', default=None, help='Optional XSUB bind address override')
    parser.add_argument('--no-kill-stale', action='store_true', help='Disable stale-instance cleanup on startup')
    args, _unknown = parser.parse_known_args()

    if args.name is None:
        raise ValueError("Missing required process name argument (e.g. Servers/Imagehub)")

    run_server(
        args.name,
        pubbinding_override=args.pub,
        subbinding_override=args.sub,
        kill_stale=not args.no_kill_stale,
    )
