
import numpy as np
from pytweezer.servers import DataClient,ImageClient
from pytweezer.servers import *
import argparse
from imageio import imread

class ImageSubstract():
    _imagestreams = PropertyAttribute('imagestreams',['None'])
    _background_path = PropertyAttribute('background_path', '[]')
    _bit_resolution_bg = PropertyAttribute('bit_resolution_bg', 16)
    _bit_resolution_im = PropertyAttribute('bit_resolution_im', 8)


    def __init__(self,name):
        self.name = name
        self._props=Properties(name)

        self.imageq=ImageClient(name.split('/')[-1])
        self.imageq.subscribe(self._imagestreams)
        self.background = imread(self._background_path).astype(np.float64)
        if self._bit_resolution_bg != self._bit_resolution_im:
            self.background /= 2**(self._bit_resolution_bg-self._bit_resolution_im)
        print(self.background.dtype)
        print('substract_image.py subscriptions: ',self._imagestreams)

    def run(self):
        prop=self._props
        while True:
            msg=self.imageq.recv()
            if msg!= None:
                msgstr,head,img=msg

                if img.shape != self.background.shape:
                    if img.shape == self.background.T.shape:
                        self.background = self.background.T
                img_bfree = img - self.background
                self.imageq.send(np.abs(img_bfree),head,channel='_')


def main_run(name):
    sub = ImageSubstract(name)
    sub.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name=args.name[0]
    main_run(name)
