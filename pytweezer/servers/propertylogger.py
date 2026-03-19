from pytweezer.servers.properties import Properties
from pytweezer.servers import configreader  as cr
from pytweezer.servers import zmqcontext
import time
import argparse
import json
import zmq

def run_propertylogger(name):
    conf=cr.Config()
    p=Properties(name,initfromfile=True)
    rep_socket = zmqcontext.socket(zmq.REP)
    #rep_socket.RCVTIMEO = p.get('reptimeout',100)
    rep_socket.bind(conf['Servers']['Propertylogger']['rep'])
    lasttime=0
    check_interval = p.get('/Servers/ProperyLogger/check_interval', 10)
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


if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    parser.add_argument('name', nargs=1, help='name of this program instance')
#    args = parser.parse_args()
#    name=args.name[0]
    name='Servers/ProperyLogger'
    run_propertylogger(name)

