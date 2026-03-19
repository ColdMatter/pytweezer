
import numpy as np
from sympy.stats.sampling.sample_numpy import numpy

from pytweezer.servers import DataClient, ImageClient
from pytweezer.servers import Properties, PropertyAttribute
import argparse
from pytweezer.analysis.print_messages import print_error
import pandas as pd
import skimage.feature as sk


class ablation_scan_analysis:
    _datastreams = PropertyAttribute('datastreams', ['Experiment.start'])
    _imagestreams   =PropertyAttribute('imagestreams',['None'])
    _ionindex  = PropertyAttribute('checkIonImage',0)
    _survindex    = PropertyAttribute('ionSurvivalImage',1)
    _hotindex = PropertyAttribute('hotIonImage',2)
    _shelvedindex = PropertyAttribute('shelvedIonImage',3)
    _min_sigma      =PropertyAttribute('min_sigma',4.0)
    _max_sigma      =PropertyAttribute('max_sigma',30.0)
    _threshold      =PropertyAttribute('threshold',10.0)
    _filter_by_ROI = PropertyAttribute('Filter by ROI', False)
    _roi_name = PropertyAttribute('Region_of_Interest','/ROI/name')
    _verbose_output = PropertyAttribute('verbose_output', False)

    def __init__(self, name):
        self.name = name
        self._props = Properties(name)
        self.dataq = DataClient(name.split('/')[-1])
        self.dataq.subscribe(self._datastreams)
        self.imageq = ImageClient(name.split('/')[-1])
        self.imageq.subscribe(self._imagestreams)
        self.experiment = None
        print('ablation_scan_analysis.py subscriptions: ', self._imagestreams)
        print('ablation scan initialized')
        #for ds in self._datastreams:
        #    self.dataq.subscribe([ds])
        #    if self._verbose_output:
        #        print('ablation_mirror.py: subscribed to', ds)

        self._reset()

    def _reset(self):
        if self._verbose_output:
            print_error('ablation_scan_analysis.py: reset dataframe', t= 'warning')
        self.results = pd.DataFrame(columns=['task', 'rep', 'run', 'image_index', 'head', 'rf_voltage', 'shutter_time_Radba', 'rf_trap_depth', 'ba493_ion_cooling_power', 'timestamp', 'N_bright', 'loading_attempts', 'Nr_Pulses', 'mirror_position'])
        self.attempt_counter = 0
    def run(self):
        while True:
            msg_dat = self.dataq.recv()
            #print('dataq received:', msg_dat)
            msg_im = self.imageq.recv()
            #print('imageq received:', msg_im)
            if True:
                if msg_dat is not None:
                    self._reset()
                    if self._verbose_output:
                        print_error(f'ablation_scan_analysis.py: msg_dat: {msg_dat}', t='info')
                    if len(msg_dat) == 2:
                        #print('msg_dat:', msg_dat)
                        #print()
                        _, some_experiment = msg_dat
                        #print(experiment)
                        #print('')
                        if some_experiment['_name'] == 'ablation_mirror':
                            self.experiment = some_experiment
                    else:
                        print_error('ablation_scan_analysis.py: Got invalid message length {0}'.format(len(msg_dat)),
                                    'error')
                # image analysis
                if msg_im is not None:
                    msgstr,head_im,img = msg_im
                    # Check whether image belongs to experiment
                        # Use task, run, rep from both messages (data, image)
                        # If image doesn't fit to experiment, reset analysis script
                    # check that the image corresponds to the experiment
                    if self.experiment is not None:
                        if head_im['_task'] == self.experiment['_task'] and head_im['_run'] == self.experiment['_run'] and head_im['_repetition'] == self.experiment['_repetition']:
                            if self._verbose_output:
                                print_error('ablation_scan_analysis.py: image corresponds to experiment', 'info')
                            scale = head_im['_imgresolution']
                            offs = head_im['_offset']
                            imageindex = head_im['_imgindex']
                            bloblist = sk.blob_dog(img.astype(float), min_sigma=self._min_sigma, max_sigma=self._max_sigma,
                                                   threshold=self._threshold)
                            # eventually filter out blobs that are outside the ROI
                            if self._filter_by_ROI:
                                # get position and size of the ROI
                                roi_position = self._props.get(self._roi_name + '/pos', [0, 0])
                                roi_size = self._props.get(self._roi_name + '/size', [img.shape])

                                # calculate the indices of the lower left edge of the ROI
                                pos_idx = np.array([int(i / scale[n] - offs[n]) for n, i in enumerate(roi_position)])

                                # calculate the indices of the upper right edge of the ROI
                                size_idx = np.array([int(i / scale[n]) for n, i in enumerate(roi_size)])
                                pos2_idx = pos_idx + size_idx

                                # check whether the index vectors of the blobs lie inside the ROI
                                is_inside = np.logical_and(bloblist[:, :2] > pos_idx, bloblist[:, :2] < pos2_idx)

                                # if either component of the vector is False (outside ROI) we want to filter it out
                                is_inside = np.logical_and(is_inside[:, 0], is_inside[:, 1])
                                bloblist = bloblist[is_inside]

                            # create the image mask (0=transparent, 1=opaque)
                            mask = np.zeros(img.shape)

                            for blob in bloblist:
                                # Check without try/except whether blob is inside image
                                try:
                                    mask[int(blob[0]), int(blob[1]):int(blob[1]) + 5] = 1
                                except:
                                    print('ablation_scan_analysis.py: blob detect out of range')
                            N_bright = len(bloblist)
                            print_error(f'bloblist: {bloblist} \tN_bright: {N_bright}', t= 'info')
                            head_im.update(self.experiment)
                            head = head_im
                            head['N_bright'] = N_bright
                            head['ablation_bloblist'] = np.array_repr(bloblist)
                            if self._verbose_output:
                                print(f'head that is send to dataq: {head}')
                            self.dataq.send(head)
                        else:
                            self._reset()
                            if self._verbose_output:
                                print_error('ablation_scan_analysis.py: got wrong picture reset dataframe', 'warning')


def main_run(name):
    analysis = ablation_scan_analysis(name)
    analysis.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name = args.name[0]
    main_run(name)