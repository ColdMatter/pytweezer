''' Select one single image per shot from the stream using the imgindex.

Input:
    One image stream

Output:
    Imagestream, the selected image

Properties:
    *   imagestreams: ([str]) input image streams
    *   SelectedImage: (int) the number of image to be selected counting from 0.


'''
import numpy as np
from pytweezer.servers import DataClient,ImageClient
from pytweezer.servers import Properties,PropertyAttribute
import argparse
class ImageSlice():

    _imagestreams   =PropertyAttribute('imagestreams',['None'])
    _index      =PropertyAttribute('SelectedImage',1)

    def __init__(self,name):
        self._props=Properties(name)
        self._name=name


        self.imageq=ImageClient(name.split('/')[-1])
        self.imageq.subscribe(self._imagestreams)
        print('imageselector.py subscriptions: ',self._imagestreams)

    def run(self):
        while True:
            msg=self.imageq.recv()
            if msg!= None:
                msgstr,head,img=msg

                if head['_imgindex']==self._index:
                    self.imageq.send(img,head,channel='_')


def main_run(name):
    slc=ImageSlice(name)
    slc.run()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name=args.name[0]
    main_run(name)
