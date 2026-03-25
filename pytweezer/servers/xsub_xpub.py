from typing import Any

import argparse
import os
import threading

import zmq
from zmq.utils.monitor import recv_monitor_message

from pytweezer.servers.configreader import ConfigReader

EVENT_MAP = {}
# print("Event names:")
for name in dir(zmq):
    if name.startswith('EVENT_'):
        value = getattr(zmq, name)
        # print("%21s : %4i" % (name, value))
        EVENT_MAP[value] = name


def _resolve_bind_endpoint(endpoint: str, bind_ip: str | None) -> str:
    """
    Replace the host in a tcp://host:port endpoint with bind_ip.

    Example:
        endpoint = tcp://localhost:1114
        bind_ip  = 0.0.0.0
        result   = tcp://0.0.0.0:1114
    """
    if bind_ip is None or not endpoint.startswith("tcp://"):
        return endpoint
    _prefix, port = endpoint.rsplit(":", 1)
    return f"tcp://{bind_ip}:{port}"


def run_server(name, bind_ip=None):
    if bind_ip is None:
        bind_ip = os.getenv('PYTWEEZER_BIND_IP')
    conf = ConfigReader.getConfiguration()
    name = name.split('/')[-1]
    pubbinding = _resolve_bind_endpoint(conf['Servers'][name]['pub'], bind_ip)
    subbinding = _resolve_bind_endpoint(conf['Servers'][name]['sub'], bind_ip)
    #print(name + " starting on")
    #print('XSUB: ', subbinding)
    #print('XPUB: ', pubbinding)
    context = zmq.Context()
    sock_sub = context.socket(zmq.XSUB)
    sock_pub = context.socket(zmq.XPUB)
    sock_sub.setsockopt(zmq.LINGER, 0)
    sock_pub.setsockopt(zmq.LINGER, 0)
    sock_pub.setsockopt(zmq.XPUB_VERBOSE, 1)
    sock_sub.bind(subbinding)
    sock_pub.bind(pubbinding)

    pub_mon = sock_pub.get_monitor_socket()
    pub_mon_thread = threading.Thread(
        target=event_monitor, args=(pub_mon, name, 'PUB'), daemon=True
    )
    pub_mon_thread.start()
    sub_mon = sock_sub.get_monitor_socket()
    sub_mon_thread = threading.Thread(
        target=event_monitor, args=(sub_mon, name, 'SUB'), daemon=True
    )
    sub_mon_thread.start()

    try:
        # run server forever
        zmq.proxy(sock_sub, sock_pub)
    finally:
        sock_sub.close()
        sock_pub.close()
        context.term()


def event_monitor(monitor: zmq.Socket, name, pub_sub) -> None:
    while True:
        if not monitor.poll(timeout=1000):
            continue
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
    parser.add_argument('name', nargs=1, help='name of this program instance')
    parser.add_argument(
        '--bind-ip',
        default=None,
        help='optional bind host override for tcp endpoints, e.g. 0.0.0.0',
    )
    args = parser.parse_args()
    name = args.name[0]
    run_server(name, bind_ip=args.bind_ip)
