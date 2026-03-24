from typing import Any

import zmq
import sys

sys.path.append('../../')
sys.path.append('../')
from pytweezer import *
import json
import argparse
from pytweezer.servers.configreader import ConfigReader
import threading
from zmq.utils.monitor import recv_monitor_message
from pytweezer.analysis.print_messages import print_error
import time

EVENT_MAP = {}
# print("Event names:")
for name in dir(zmq):
    if name.startswith('EVENT_'):
        value = getattr(zmq, name)
        # print("%21s : %4i" % (name, value))
        EVENT_MAP[value] = name


def run_server(name):
    conf = ConfigReader.getConfiguration()
    name = name.split('/')[-1]
    pubbinding = conf['Servers'][name]['pub']
    subbinding = conf['Servers'][name]['sub']
    #print(name + " starting on")
    #print('XSUB: ', subbinding)
    #print('XPUB: ', pubbinding)
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
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name = args.name[0]
    run_server(name)
