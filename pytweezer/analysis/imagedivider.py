'''Handle absorption imaging data.

According to Beers Law, light travelling through an atomic cloud is attenuated by:

    :math:`I=I_0e^{-n(x,y)\\sigma}`

where :math:`I_0` is the intensity before the cloud (Imageing beam without atoms) and :math:`I` is the
attenuated intensity (Image with atoms) :math:`\\sigma` is the absorption cross section and n(x,y) is the
column density of the atoms. Transorming this equation allows us to determine the column density of the atoms
from the import two images.

:math:`n(x,y)=\\frac{ln(I_0/I)}{\\sigma}`

where :math:`\\sigma` is given by:



:math:`N_{atoms}=\\frac{ln(I_0/I)}{\\frac{\\sigma_0}{1+(2\\Delta/\\Gamma)^2}}`

where effscale recsales the value from atoms per :math:`m^2` to atoms per pixel.


:math:`\\sigma_0=\\frac{\\hbar\\omega\\Gamma}{2 I_{sat}/C_2}`

where :math:`\\omega` is the angular frequency of the laser, :math:`\\Gamma` is the linewidth of the
transition in angular frequencies, :math:`I_{sat}` is the saturation intensity in SI units. and :math:`C_2` is
the Clebsh Gordon coefficient for the dipole transition.

Properties:

    *   imagestreams:
        input datastreams
    *   imageWithAtoms: index of the image containing the imagedata with atoms (0  being the first image in
        each run)
    *   imageWithoutAtoms:  the image containing the image without atoms
    *   absorptionCrossection:  defaults to :math:`\\sigma_0`
    *   detuning_rad:    detuning of the imaging laser from resonance
    *   linewidth_rad:   linewidth of the transition


'''


import numpy as np
from pytweezer.servers import DataClient,ImageClient
from pytweezer.servers import Properties,PropertyAttribute
import argparse


class ImageDivider():

    _imagestreams   =PropertyAttribute('imagestreams', ['None'])
    _flushindex     =PropertyAttribute('flushImage', 0)
    _atomindex      =PropertyAttribute('imageWithAtoms', 1)    # needs to be lower than the image without atoms
    _noatomindex    =PropertyAttribute('imageWithoutAtoms', 2)
    _backgroundindex=PropertyAttribute('imageBackground', 3)
    _selectindex    =PropertyAttribute('selectedImage', 1)
    _verbose_output = PropertyAttribute('verbose_output', False)
    isat=2.54e1     # saturation intensity w/m2. this equals 2.54mW/cm**2. Meaning we are at I << I sat
    h_bar = 1.0545716346179718e-34
    omega=446.799677E12*2*np.pi
    gamma=5.8724*1E6*2*np.pi
    # We use linear polarized light perpendicular to the quantization axis. The cross section
    # therefore reduces to 1/2 as the linear light is projected on the circular sigma plus and minus
    # eigenbasis, given by the quantization axis and the fact that only half of the light (sigma minus)
    # is absorbed.
    C_2 = 1 # 1./2.
    sig0 = h_bar*omega*gamma/2/isat*C_2
    _sigma0         = sig0  # PropertyAttribute('absorptionCrossection', sig0)
    _det            = PropertyAttribute('detuning_rad', 0)
    _gamma          = PropertyAttribute('linewidth_rad', gamma)

    def __init__(self,name):
        self._props=Properties(name)
        self._name=name

        self.image_detuning = None
        self.nextimage = self._atomindex
        self.gotAImage = False
        self.gotNAImage = False
        self.gotBGImage = False

        self.dataq=DataClient(name.split('/')[-1])
        self.dataq.subscribe(['Experiment.start'])
        self.imageq=ImageClient(name.split('/')[-1])
        self.imageq.subscribe(self._imagestreams)
        print('imagedivider.py subscriptions:', self._imagestreams)

    def run(self):
        while True:
            msg = self.imageq.recv()
            if msg != None:
                msgstr, head, img = msg
                img = img.astype(np.float)
                if self._verbose_output:
                    print('imagedivider.py: np.max(img) =', np.max(img))
                    print('imagedivider.py: np.mean(img) =', np.mean(img))

                # if selected image arrived store the image
                if head['_imgindex'] == self._selectindex:
                    # additional checking for _repetition, _run,_task might prevent further errors
                    self.sImage = img
                    self.sImageHead = head

                if head['_imgindex'] == self._flushindex and not self._flushindex == self._atomindex:
                    if self._verbose_output:
                        print('imagedivider.py: got flush image')

                #if image with atoms arrived store the image
                if head['_imgindex'] == self._atomindex:
                    # additional checking for _repetition, _run,_task might prevent further errors
                    self.gotAImage = True
                    self.aImage = img
                    self.aImageHead = head
                    self.scale = head['_imgresolution']
                    # self.effscale=self.scale[0]*self.scale[1]   #m2 per pixel
                    if self._verbose_output:
                        print('imagedivider.py: got atoms image')
                    self.update_experiment_parameters()
                    # print('imagedivider.py: current detuning =', self.image_detuning)
                # if the second image without the atoms arrived perform evaluation
                elif head['_imgindex']==self._noatomindex:
                    self.gotNAImage=True
                    self.nAImage = img
                    self.head = head
                    if self._verbose_output:
                        print('imagedivider.py: got noatoms image')

                # if the background image (the third image) arrived perform the evaluation
                elif head['_imgindex'] == self._backgroundindex:
                    self.gotBGImage = True
                    self.BGImage = img
                    self.head = head
                    if self._verbose_output:
                        print('imagedivider.py: got background image')

                if self.gotNAImage and self.gotAImage and self.gotBGImage:
                    self._evaluate_and_send()
                if self._verbose_output:
                    print('imagedivider.py: head[\'_imgindex\'] =', head['_imgindex'])

    def _evaluate_and_send(self):
        if self._verbose_output:
            print('imagedivider.py: send2')
        # delta=np.abs(2*np.pi*(self.image_detuning-1.6)*1E6/(self.gamma/2))
        delta = np.abs(self._det/(self._gamma/2))
        if self._verbose_output:
            print('imagedivider.py: Delta in nat lw =', delta)
        no_atoms_bfree = self.nAImage - self.BGImage
        atoms_bfree = self.aImage - self.BGImage
        # mask out all pixels with negative values
        no_atoms_bfree_m = np.ma.MaskedArray(no_atoms_bfree, no_atoms_bfree <= 0)
        atoms_bfree_m = np.ma.MaskedArray(atoms_bfree, atoms_bfree <= 0)
        quotient = no_atoms_bfree/atoms_bfree
        quotient_m = np.ma.MaskedArray(quotient, quotient <= 0)

        # calculate number of invalid pixels and maximum attenuation
        # att_max = np.max(quotient)

        # print('imagedivider.py: Maximum attenuation = {:.1f}%'.format(att_max*100))

        img = np.log(quotient_m) * (1 + delta ** 2) / self._sigma0
        n_invalid = img.mask.sum()

        img = np.ma.fix_invalid(img, img.mask, fill_value=0)
        if self._verbose_output:
            print('imagedivider.py: Number of invalid pixels: {:.0f}'.format(n_invalid))
        # img=np.log(quotient)*(1)/self._sigma0
        # img=np.log((self.nAImage)/(self.aImage))*(1+delta**2)/self._sigma0
        self.imageq.send(img, self.head, channel='_')
        self.imageq.send(self.sImage, self.sImageHead, channel='Atoms')

        self.gotNAImage=False
        self.gotAImage=False
        self.gotBGImage=False

    def update_experiment_parameters(self):
        ''' keep list of experiment starts updatesd'''

        while self.dataq.has_new_data():
            # print('imagedivider.py: datamgr.py data incomming!')
            recvmsg = self.dataq.recv()
            A = None
            if len(recvmsg) == 2:
                msg, msg_dic = recvmsg
            elif len(recvmsg) == 3:
                msg, msg_dic, A = recvmsg
            else:
                print('imagedivider.py: datamgr message error')
                return
            # print('imagedivider.py: msg =', msg)
            if msg == 'Experiment.start':
                if 'image_detuning' in msg_dic:
                    self.image_detuning = msg_dic['image_detuning']
                # indices=dict((k,msg_dic[k]) for k in ('_run','_repetition','_task') if k in msg_dic)
                # timestamp=msg_dic['_starttime']
                # self.experiment_timings.append([timestamp,indices])
                # while len(self.experiment_timings)>20:
                #    del self.experiment_timings[0]


def main_run(name):
    slc = ImageDivider(name)
    slc.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name=args.name[0]
    main_run(name)
