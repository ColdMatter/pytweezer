''' Discriminates the presenence or lack of an ion in a series of images.

Intended use is determining the product state of a single ion after interaction
with atoms. For more general use, cold be adapted to look at an arbitrary number
of images.

We use skimage.features.blob_dog for bright spot detection

Input:
    Imagestream

Output:
        Datastream:
            *   ionchecklist    Truth table for presence of ion in each image
            *   result          int corresponding to ion product state

Properties:
    *   imagestreams   ([str]) Input streams
    *   max_sigma   (float)    Parameter from blob_dog
    *   threshold   (float)    Parameter from blob_dog
    *   Region_of_Interest     A region of interest that can be used to sort out unwanted blobs
    *   Filter by ROI   (bool) If True, blobs outside the ROI will be sorted out


'''

import numpy as np
from pytweezer.servers import DataClient,ImageClient
from pytweezer.servers import Properties,PropertyAttribute
import skimage.feature as sk
import argparse
import cv2

class IonState():
    _imagestreams = PropertyAttribute('imagestreams',['None'])
    _ionindex  = PropertyAttribute('checkIonImage',0)
    _survindex    = PropertyAttribute('ionSurvivalImage',1)
    _hotindex = PropertyAttribute('hotIonImage',2)
    _shelvedindex = PropertyAttribute('shelvedIonImage',3)
    _min_sigma      =PropertyAttribute('min_sigma',10.0)
    _max_sigma      =PropertyAttribute('max_sigma',30.0)
    _threshold      =PropertyAttribute('threshold',250.0)#300.0)
    _filter_by_ROI = PropertyAttribute('Filter by ROI', False)
    _roi_name = PropertyAttribute('Region_of_Interest','/ROI/name')
    _verbose_output = PropertyAttribute('verbose_output', False)

    def __init__(self,name):
        self._props=Properties(name)
        self.dataq=DataClient(name.split('/')[-1])
        self.dataq.subscribe(['Experiment.start'])
        self.imageq=ImageClient(name.split('/')[-1])
        self.imageq.subscribe(self._imagestreams)
        print('ion_state.py subscriptions: ',self._imagestreams)
        self.name = name

        #initalise indices: Ion check, Survival, Hot, Shelved
        self.nextimage=self._ionindex
        self.gotIImage=False
        self.gotSImage=False
        self.gotHImage=False
        self.gotShImage=False


    def run(self):
        '''
        Based on imagedivider.py
        '''
        while True:
            msg=self.imageq.recv()

            if msg!= None:
                msgstr,head,img=msg
                self.msg=msg
                self.head = head
                # print('ion state: got msg', msg)

                #check for ion before interaction
                if head['_imgindex']==self._ionindex:
                    #additional checking for _repetition, _run,_task might prevent further errors
                    self.gotIImage=True
                    self.iImage=img
                    self.iImageHead=head
                    self.scale=head['_imgresolution']
                    if self._verbose_output:
                        print('ion_state.py: got ion present image')

                #check for ion survival
                elif head['_imgindex']==self._survindex:
                    self.gotSImage=True
                    self.sImage=img
                    self.sImageHead=head
                    if self._verbose_output:
                        print('ion_state.py: got survival image')

                #check for hot ion
                elif head['_imgindex']==self._hotindex:
                    self.gotHImage=True
                    self.hImage=img
                    self.hImageHead=head
                    if self._verbose_output:
                        print('ion_state.py: got hot ion image')

                #check for shelved ion
                elif head['_imgindex']==self._shelvedindex:
                    self.gotShImage=True
                    self.shImage=img
                    self.shImageHead=head
                    if self._verbose_output:
                        print('ion_state.py: got shelved ion image')
                #print(head['_imgindex'])

                #once all images recieved, check each for ions
                if self.gotIImage and self.gotSImage and self.gotHImage and self.gotShImage:
                    self._evaluate_and_send()


    def _evaluate_and_send(self):
        '''
        run the ion checker, determine output state. creates a checklist showing
        the number of ions in each images, and then determines the ion product
        state
        '''
        if self._verbose_output:
            print('ion_state.py: checking ion state')
        ionChecklist = [None]*4
        imglist = [self.iImage,self.sImage,self.hImage,self.shImage]
        headlist = [self.iImageHead,self.sImageHead,self.hImageHead,self.shImageHead]
        for i,image in enumerate(imglist):
            head = headlist[i]
            ionChecklist[i] = self._check_for_ion(image,head)
        print('ion_state.py: checklist = \033[1m{0}\033[0m'.format(ionChecklist))

        result = -1
        result_interpretation = ''
        if ionChecklist[0] == 0:
            result = 0
            result_interpretation = 'No ion'
        elif ionChecklist == [1,1,1,1]:
            result = 1
            result_interpretation = 'Survival'
        elif ionChecklist == [1,0,1,1]:
            result = 2
            result_interpretation = 'Hot ion'
        elif ionChecklist == [1,0,0,1]:
            result = 3
            result_interpretation = 'Shelved ion'
        elif ionChecklist == [1,0,0,0]:
            result = 4
            result_interpretation = 'Lost ion'
        else:
            result = 0
            result_interpretation = 'Unclassified result'
        print('ion_state.py: {0}'.format(result_interpretation))

        #data output
        self.head['result'] = result
        self.head['result_interpretation'] = result_interpretation
        self.dataq.send(self.head,ionChecklist,'_ionchecklist')

        #required to reset the script for the next set of images
        self.gotSImage = False
        self.gotHImage = False
        self.gotShImage = False

    def _check_for_ion(self,img,head):
        '''
        Based on bright_spots.py. returns the ion checklist
        '''
        prop=self._props
        scale=head['_imgresolution']
        offs=head['_offset']
        bloblist=sk.blob_dog(img.astype(float), min_sigma=self._min_sigma, max_sigma=self._max_sigma, threshold=self._threshold)
        # blobs: (y, x, r)

        """params = cv2.SimpleBlobDetector_Params()
        # params.minThreshold = 10
        # params.maxThreshold = 100
        params.filterByArea = True
        params.minArea = 70
        params.filterByInertia = False
        params.filterByConvexity = False
        params.filterByCircularity = True
        params.minCircularity = .1

        img = np.where(img > 255, 255, img)
        img = np.where(img < 0, 0, img)
        img = img.astype(np.uint8)
        detector = cv2.SimpleBlobDetector_create(params)
        bloblist = detector.detect(255 - img)"""
        if self._verbose_output:
            print('ion_state.py: bloblist =', bloblist)
        # blob.pt.x@@@

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
        # if len(bloblist) != 0:
        #     ionCheck = True
        # else:
        #     ionCheck = False
        return len(bloblist)

def main_run(name):
    slc=IonState(name)
    slc.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name=args.name[0]
    main_run(name)
