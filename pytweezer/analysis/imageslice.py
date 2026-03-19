''' Create linear traces and small subregions from image stream

.. image:: /Imageslice.png

Input:
    Imagestream

Output:
    *   colsum\_ (datastream)    summation in the direction of columns  (use only row that lie within the
        region of interest. (gray area)
    *   rowsum\_ (datastream)    summation along rows
    *   colsumcut\_,rowsumcut\_  use only data within the region of interest
    *   colint,rowint,colintcut,rowintcut: Perform numeric integration (which means the sum is multiplied by
        the metric pixel size)
    *   (imagestream) : send the image within the ROI

Properties:
    *   cutimg: (bool)  send the image inside the ROI
    *   colsum: send the column sum
    *   rowsum: send the sum over the rows
    *   colsumcut,rowsumcut: (bool) send the corresponding data restricted to the region of interest
    *   colint,rowint,colintcut,rowintcut:  similar to the sum version but teking into account the scale.
            Therefore these versionc can be treated as integrals



'''


import numpy as np
from pytweezer.servers import DataClient,ImageClient
from pytweezer.servers import *
import argparse
class ImageSlice():
    _imagestreams   =PropertyAttribute('imagestreams',['None'])
    _roi_name=PropertyAttribute('Region_of_Interest','/ROI/name')
    #enabel or disable streams
    _cutimg=PropertyAttribute('cutimg',False)
    _colsum=PropertyAttribute('colsum',True)
    _rowsum=PropertyAttribute('rowsum',True)
    _rowsumcut=PropertyAttribute('rowsumcut',True)
    _colsumcut=PropertyAttribute('colsumcut',True)
    _colint=PropertyAttribute('colint',False)
    _rowint=PropertyAttribute('rowint',False)
    _colintcut=PropertyAttribute('colintcut',False)
    _rowintcut=PropertyAttribute('rowintcut',False)

    def __init__(self,name):
        self.name = name
        self._props=Properties(name)

        self.dataq=DataClient(name.split('/')[-1])
        self.imageq=ImageClient(name.split('/')[-1])
        self.imageq.subscribe(self._imagestreams)
        print('imageslice.py subscriptions: ',self._imagestreams)
    def run(self):
        prop=self._props
        while True:
            msg=self.imageq.recv()
            if msg!= None:
                msgstr,head,img=msg
                try:
                    scale=head['_imgresolution']
                    offs=head['_offset']
                except:
                    scale=[1,1]
                    offs=[0,0]

                pos=[int(i/scale[n]-offs[n]) for n,i  in enumerate(prop.get(self._roi_name+'/pos',[0,0]))]
                size=[int(i/scale[n]) for n,i in enumerate(prop.get(self._roi_name+'/size',list(img.shape)))]
                pos2=pos+size

                # assign ROI corners correctly if handles are mixed up
                lowerleft = [0,0]
                upperright = [0,0]
                lowerleft[0] = min(pos[0], pos2[0])
                lowerleft[1] = min(pos[1], pos2[1])
                upperright[0] = max(pos[0], pos2[0])
                upperright[1] = max(pos[1], pos2[1])
                pos = lowerleft
                pos2 = upperright

                colsum=img[:,pos[1]:size[1]+pos[1]].sum(axis=1)
                rowsum=img[pos[0]:size[0]+pos[0]].sum(axis=0)
                totalsum_roi=img[pos[0]:size[0]+pos[0],pos[1]:size[1]+pos[1]].sum()

                # calculate the numerical integral in both direction by multiplication with dx
                colint = colsum*scale[1]
                rowint = rowsum*scale[0]

                # vstack the resulting slices with their respective axis
                x_col = (np.arange(colsum.shape[0])+offs[0])*scale[0]
                x_row = (np.arange(rowsum.shape[0])+offs[1])*scale[1]
                csum = np.vstack([x_col, colsum])
                rsum = np.vstack([x_row, rowsum])
                cint = np.vstack([x_col,colint])
                rint = np.vstack([x_row,rowint])

                head.update({'totalsum_roi':totalsum_roi})
                head.update({'totalint_roi':totalsum_roi*scale[0]*scale[1]})

                #send row and column sum
                if self._colsum:
                    self.dataq.send(head,csum,'colsum_')
                if self._rowsum:
                    self.dataq.send(head,rsum,'rowsum_')

                #send cut trace
                if self._colsumcut:
                    self.dataq.send(head,np.copy(csum[:,pos[0]:size[0]+pos[0]]),'colsumcut_')
                if self._rowsumcut:
                    self.dataq.send(head,np.copy(rsum[:,pos[1]:size[1]+pos[1]]),'rowsumcut_')


                #send row and column integral
                if self._colint:
                    self.dataq.send(head,cint,'colint_')
                if self._rowint:
                    self.dataq.send(head,rint,'rowint_')

                #send cut trace
                if self._colintcut:
                    self.dataq.send(head,np.copy(cint[:,pos[0]:size[0]+pos[0]]),'colintcut_')
                if self._rowintcut:
                    self.dataq.send(head,np.copy(rint[:,pos[1]:size[1]+pos[1]]),'rowintcut_')

                #send subregion image
                if self._cutimg:
                    head['_offset']=[o+pos[n]  for n,o in enumerate(offs)]
                    self.imageq.send(np.copy(img[pos[0]:size[0]+pos[0],pos[1]:size[1]+pos[1]]),head)

def main_run(name):
    slc=ImageSlice(name)
    slc.run()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name=args.name[0]
    main_run(name)
