from pytweezer.servers.properties import Properties
from pytweezer.servers import configreader  as cr
from pytweezer.servers import zmqcontext
import time
import argparse
import json
import zmq
import os
import signal


def _resolve_server_name(name: str, conf: dict) -> str:
    if isinstance(name, str) and name.startswith('Servers/'):
        return name.split('/', 1)[1]
    if isinstance(name, str) and name in conf.get('Servers', {}):
        return name
    return 'Propertylogger'


def _terminate_stale_propertylogger_instances(instance_name: str, grace_s: float = 1.0) -> None:
    if os.name != 'posix':
        return

    this_pid = os.getpid()
    stale_pids = []
    try:
        for pid_str in os.listdir('/proc'):
            if not pid_str.isdigit():
                
                continue
            pid = int(pid_str)
            if pid == this_pid:
                continue

            cmdline_path = f'/proc/{pid_str}/cmdline'
            try:
                with open(cmdline_path, 'rb') as f:
                    cmdline = f.read().replace(b'\x00', b' ').decode('utf-8', errors='ignore')
            except Exception:
                continue

            if 'propertylogger.py' in cmdline and instance_name in cmdline:
                stale_pids.append(pid)
    except Exception:
        return

    for pid in stale_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            pass

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
        except Exception:
            pass


def run_propertylogger(name='Servers/Propertylogger'):
    conf = cr.Config()
    server_name = _resolve_server_name(name, conf)
    _terminate_stale_propertylogger_instances(server_name)
    p = Properties(f'Servers/{server_name}', initfromfile=True)

    rep_socket = zmqcontext.socket(zmq.REP)
    rep_socket.setsockopt(zmq.LINGER, 0)
    #rep_socket.RCVTIMEO = p.get('reptimeout',100)
    host = conf['Servers'][server_name].get('host', 'localhost')
    port = conf['Servers'][server_name].get('port', 3106)
    rep_endpoint = f"tcp://{host}:{port}"
    if isinstance(rep_endpoint, str) and rep_endpoint.startswith('tcp://'):
        rep_socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
        if hasattr(zmq, 'TCP_KEEPALIVE_IDLE'):
            rep_socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
        if hasattr(zmq, 'TCP_KEEPALIVE_INTVL'):
            rep_socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 30)
        if hasattr(zmq, 'TCP_KEEPALIVE_CNT'):
            rep_socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, 5)
    rep_socket.bind(rep_endpoint)
    poller = zmq.Poller()
    poller.register(rep_socket, zmq.POLLIN)
    lasttime=0
    poll_timeout_ms = int(p.get(f'/Servers/{server_name}/check_interval', 10) * 1000)
    while True:
        #handle requests
        events = dict(poller.poll(poll_timeout_ms))
        if rep_socket in events and events[rep_socket] == zmq.POLLIN:
            message=rep_socket.recv_string()
            if message == 'INIT?':
                #print('propertylogger.py: someone wants data',message)        
                rep_socket.send_string('INIT?',zmq.SNDMORE)
                rep_socket.send_json(p.get('/',{}))
            elif message is not None:
                print('propertylogger.py unknown message: ',message)

        #do logging
        t = time.time()
        if t - lasttime > p.get('logginginterval[s]',1):
            lasttime = t
            #print(cr.propertyfilename)
            with open(cr.propertyfilename, "w") as outfile:
                json.dump(p.get('/', {}), outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs='?', default='Servers/Propertylogger', help='name of this program instance')
    args, _unknown = parser.parse_known_args()
    run_propertylogger(args.name)

