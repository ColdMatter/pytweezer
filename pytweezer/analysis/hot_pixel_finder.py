''' Filters out hot pixels from an image, similar to a median filter with a threshold

TODO: documentation
'''

# build and import cython gauss fit routine
import pyximport
pyximport.install()
from pytweezer.analysis.src.gaussskewfit import gaussian_fit_skewed

import numpy as np
import argparse
from pytweezer.servers import DataClient,ImageClient,Properties,PropertyAttribute


class HotFilter:
    _imagestreams   =PropertyAttribute('imagestreams',['None'])
    _threshold      =PropertyAttribute('threshold',30)
    _hotpixels      =PropertyAttribute('hotpixels', [])


    def __init__(self,name):
        self.name = name
        self._props=Properties(name)

        self.dataq=DataClient(name.split('/')[-1])
        self.imageq=ImageClient(name.split('/')[-1])
        self.imageq.subscribe(self._imagestreams)
        print('imageslice.py subscriptions: ',self._imagestreams)

    def run(self):
        prop=self._props
        prop.set('/Cameras/Analysis/' + self._imagestreams[0] + '/hotpixels', [])
        waiting = True
        while waiting:
            msg = self.imageq.recv()
            #print(waiting)
            if msg!= None:
                print('hot pixel finder image received')
                msgstr,head,img=msg
                try:
                    scale=head['_imgresolution']
                    offs=head['_offset']
                except:
                    scale=[1,1]
                    offs=[0,0]
                filteredimg = img.copy()
                for i,row in enumerate(img):
                    for j,pixel in enumerate(row):
                        comp = []
                        if j != len(row)-1:  comp.append(img[i,j+1])
                        if j != 0: comp.append(img[i, j-1])
                        if i != len(img)-1: comp.append(img[i+1, j])
                        if i != 0: comp.append(img[i-1, j])
                        comp = np.array(comp)
                        if all(pixel > self._threshold*comp):
                            print('Hot pixel at: [{},{}]'.format(i, j))
                            print('Pixel value = {}'.format(pixel))
                            print('Surrounding pixels: {}'.format(comp))
                            hotpix = prop.get('/Cameras/Analysis/'+self._imagestreams[0]+'/hotpixels',[])
                            hotpix.append((i,j))
                            prop.set('/Cameras/Analysis/' + self._imagestreams[0] + '/hotpixels', hotpix)


                    self.imageq.send(filteredimg, head, channel='_')
                waiting = False
def main_run(name):
    slc=HotFilter(name)
    slc.run()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name=args.name[0]
    main_run(name)