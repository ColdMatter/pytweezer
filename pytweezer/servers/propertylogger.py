from pytweezer.servers.properties import Properties
from pytweezer.servers import configreader  as cr
import time
import argparse
import json
import zmq
import os


def _resolve_bind_endpoint(endpoint: str, bind_ip: str | None) -> str:
    if bind_ip is None or not endpoint.startswith('tcp://'):
        return endpoint
    _prefix, port = endpoint.rsplit(':', 1)
    return f"tcp://{bind_ip}:{port}"

def run_propertylogger(name, bind_ip=None):
    if bind_ip is None:
        bind_ip = os.getenv('PYTWEEZER_BIND_IP')
    conf=cr.Config()
    p=Properties(name,initfromfile=True)
    context = zmq.Context()
    rep_socket = context.socket(zmq.REP)
    rep_socket.setsockopt(zmq.LINGER, 0)
    #rep_socket.RCVTIMEO = p.get('reptimeout',100)
    rep_socket.bind(_resolve_bind_endpoint(conf['Servers']['Propertylogger']['rep'], bind_ip))
    lasttime=0
    check_interval = p.get('/Servers/ProperyLogger/check_interval', 10)
    try:
        while True:
            #handle requests
            time.sleep(0.01)
            if rep_socket.poll(1)==zmq.POLLIN:
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
                    json.dump(p.properties, outfile, indent=4)
    finally:
        rep_socket.close()
        context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('instance_name', nargs='?', default=None)
    parser.add_argument('--name', default='Servers/ProperyLogger')
    parser.add_argument(
        '--bind-ip',
        default=None,
        help='optional bind host override for tcp endpoint, e.g. 0.0.0.0',
    )
    args = parser.parse_args()
    run_name = args.name
    if args.instance_name:
        run_name = args.instance_name
    run_propertylogger(run_name, bind_ip=args.bind_ip)

