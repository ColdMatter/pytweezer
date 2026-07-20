import time
import numpy as np
from pytweezer.servers import DataClient,ImageClient
from pytweezer.servers import *
import argparse
class DataSubstract():
    """
        This analysis script substracts a fixed data array from an incoming datastream.

    """
    _datastreams = PropertyAttribute('datastreams',['None'])
    _background = PropertyAttribute('background', '[]')

    def __init__(self, name):
        self.name = name
        self._props = Properties(name)

        self.dataq = DataClient(name.split('/')[-1])
        self.dataq.subscribe(self._datastreams)
        print('data_average.py subscriptions: ',self._datastreams)

    def run(self):
        while True:
            background = np.array(self._background, dtype=np.float)
            # check if there is new data in the data queue
            if self.dataq.has_new_data():
                msg = self.dataq.recv()
                if msg!= None:
                    msgstr, head, data = msg
                    data = np.copy(data)
                else:
                    continue
            else:
                continue
            # if we have valid data, try to substract background and publish
            try:
                data[1] -= background[1]
                self.dataq.send(head, data, '')
            except ValueError:
                print(self.name, ": Data and background ain't shape matched.")


def main_run(name):
    bgs = DataSubstract(name)
    bgs.run()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name=args.name[0]
    main_run(name)
