
from pytweezer.servers import Properties,ImageClient
from pytweezer.servers import configreader  as cr
import time
import argparse
import json




class ImageStreamLogger():
    """ monitors the image stream and provides information """
    def __init__(self,name):
        self.prop=Properties(name)
        self.stream=ImageClient(name)
        self.stream.subscribe('')
    def run(self):
        ''' detect active channels'''
        lastupdate=time.time()
        active=self.prop.get('active',{})
        while True:
            msg=self.stream.recv()
            updated=False
            if msg != None:
                active[msg[0]]={'timestamp':time.time(),'parts':len(msg)}
                updated=True
            if (time.time()-lastupdate>1) and (updated): #update dictionary every second only
                #print(type(active))
                #print(time.time())
                self.prop.set("active",active)
                updated=False
                lastupdate=time.time()


def run_ImageLogger(name):
    d=ImageStreamLogger(name)
    d.run()



if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    parser.add_argument('name', nargs=1, help='name of this program instance')
#    args = parser.parse_args()
#    name=args.name[0]
    name='Servers/ImageStream'
    run_ImageLogger(name)
