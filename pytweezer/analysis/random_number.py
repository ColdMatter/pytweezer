'''

random integers for testing

'''

import numpy as np
from pytweezer.servers import DataClient,ImageClient
from pytweezer.servers import Properties,PropertyAttribute
import argparse
from random import randint

class RandomNumber():
    _max = PropertyAttribute('max',1)

    def __init__(self,name):
        self._props=Properties(name)
        self.dataq=DataClient(name.split('/')[-1])
        self.dataq.subscribe(['Experiment.start'])
        self.name = name



    def run(self):
        '''
        Based on imagedivider.py
        '''
        while True:
            msg = self.dataq.recv()
            # print('gaussfit.py recv')
            if msg != None:
                # print(name+'received')
                msg, head = msg
                n = randint(0,self._max)
                print('random number:', n)
                data = {'n':n}
                self.dataq.send(data, '_n')


def main_run(name):
    slc=RandomNumber(name)
    slc.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name=args.name[0]
    main_run(name)
