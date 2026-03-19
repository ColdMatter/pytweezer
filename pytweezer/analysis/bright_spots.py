''' Detect bright spots such a ions in an image. 
We use skimage.features.blob_dog for bright spot detection

Input:
    Imagestream

Output:
    *   Imagestream:
            *   np.array    Mask containing ones on the locations where the bright spots are
            *   dict        N_bright the number of bright spots detected
        Datastream:
            *   bloblist    the locations of the pright spots
            *   dict        N_bright the number of bright spots detected

Properties:
    *   imagestreams   ([str]) Input streams
    *   max_sigma   (float)    Parameter from blob_dog
    *   threshold   (float)    Parameter from blob_dog
    *   Region_of_Interest     A region of interest that can be used to sort out unwanted blobs
    *   Filter by ROI   (bool) If True, blobs outside the ROI will be sorted out

Hint:
    Use the output stream as a mask in the image monitor


'''
import numpy as np
from pytweezer.servers import DataClient,ImageClient
from pytweezer.servers import Properties,PropertyAttribute
import argparse
import skimage.feature as sk
class BrightSpots():
    _imagestreams = PropertyAttribute('imagestreams',['None'])
    _ionindex  = PropertyAttribute('checkIonImage',0)
    _survindex    = PropertyAttribute('ionSurvivalImage',1)
    _hotindex = PropertyAttribute('hotIonImage',2)
    _shelvedindex = PropertyAttribute('shelvedIonImage',3)
    _min_sigma      =PropertyAttribute('min_sigma',10.0)
    _max_sigma      =PropertyAttribute('max_sigma',30.0)
    _threshold      =PropertyAttribute('threshold',300.0)
    _filter_by_ROI = PropertyAttribute('Filter by ROI', False)
    _roi_name = PropertyAttribute('Region_of_Interest','/ROI/name')
    _verbose_output = PropertyAttribute('verbose_output', False)

    def __init__(self,name):
        self._props=Properties(name)
        self.dataq=DataClient(name.split('/')[-1])
        self.imageq=ImageClient(name.split('/')[-1])
        self.imageq.subscribe(self._imagestreams)
        self.name = name
        #print(name)


    def run(self):
        prop=self._props
        while True:
            msg=self.imageq.recv()
            if msg!= None:
                msgstr,head,img=msg
                scale=head['_imgresolution']
                offs=head['_offset']
                bloblist = sk.blob_dog(img.astype(float), min_sigma=self._min_sigma, max_sigma=self._max_sigma,
                                       threshold=self._threshold)
                # eventually filter out blobs that are outside the ROI
                if self._filter_by_ROI:
                    # get position and size of the ROI
                    roi_position = prop.get(self._roi_name + '/pos', [0,0])
                    roi_size = prop.get(self._roi_name + '/size', [img.shape])

                    # calculate the indices of the lower left edge of the ROI
                    pos_idx = np.array([int(i/scale[n]-offs[n]) for n,i  in enumerate(roi_position)])

                    # calculate the indices of the upper right edge of the ROI
                    size_idx = np.array([int(i/scale[n]) for n,i in enumerate(roi_size)])
                    pos2_idx = pos_idx + size_idx

                    # check whether the index vectors of the blobs lie inside the ROI
                    is_inside = np.logical_and(bloblist[:,:2]>pos_idx, bloblist[:,:2]<pos2_idx)

                    # if either component of the vector is False (outside ROI) we want to filter it out
                    is_inside = np.logical_and(is_inside[:,0], is_inside[:,1])
                    bloblist = bloblist[is_inside]

                # create the image mask (0=transparent, 1=opaque)
                mask=np.zeros(img.shape)

                for blob in bloblist:
                    try:
                        mask[int(blob[0]),int(blob[1]):int(blob[1])+5] = 1
                    except:
                        print('bright_spots.py: blob detect out of range')
                head['N_bright']=len(bloblist)
                if self._verbose_output:
                    print('bright_spots.py:', bloblist, '\tN_bright:', head['N_bright'])

                self.imageq.send(mask,head)
                self.dataq.send(head,bloblist,'bloblist_')

def main_run(name):
    slc=BrightSpots(name)
    slc.run()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name=args.name[0]
    main_run(name)









