import numpy as np
from pytweezer.servers import DataClient,ImageClient
from pytweezer.servers import *
import argparse
from imageio import imread
import imutils

class ImageRotate():
    _imagestreams = PropertyAttribute('imagestreams',['None'])
    _angle = PropertyAttribute('angle', 0.0)


    def __init__(self,name):
        self.name = name
        self._props = Properties(name)

        self.imageq = ImageClient(name.split('/')[-1])
        self.imageq.subscribe(self._imagestreams)
        print('rotate_image.py subscriptions: ',self._imagestreams)

    def run(self):
        prop=self._props
        while True:
            msg=self.imageq.recv()
            if msg!= None:
                msgstr,head,img=msg

                # rotate the image
                img_rot = imutils.rotate(img, self._angle)

                # rotate the coordinate system
                offsetx, offsety = head["_offset"]
                offxrot = np.sin(np.deg2rad(self._angle)) * offsetx + np.cos(np.deg2rad(self._angle)) * offsety
                offyrot = np.sin(np.deg2rad(self._angle)) * offsety+ np.cos(np.deg2rad(self._angle)) * offsetx
                head["_offset"] = [offxrot, offyrot]

                # send the image
                self.imageq.send(np.abs(img_rot),head,channel='_')


def main_run(name):
    sub = ImageRotate(name)
    sub.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='rotate_image.py')
    args = parser.parse_args()
    name=args.name[0]
    main_run(name)
