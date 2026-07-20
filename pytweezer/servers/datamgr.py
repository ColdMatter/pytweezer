import datetime

from pytweezer.servers import Properties,DataClient,ImageClient
import time
import numpy as np
import h5py
from pytweezer.analysis.print_messages import print_error
import pandas as pd

class DataSummary:
    """ summarize dataflow of multiple input channels

        incoming data is accumulated in self.incompleteData as soon as data from all subscribing channels
        has arrived the data is appended to the complete array. In case some data has not arrived timeout
        seconds after the end of the run the missing data will be replaced by zeros

    """

    data_update_checks = 0  # static variable
    got_data = False

    def __init__(self,name,props=None):
        if props is None:
            props = Properties(name)
        self.name = name
        self.props = props
        self.incompleteData = []
        self.completeData = []
        self.streamlist=[]
        self.task = -1
        self.dataq = DataClient(self.name)
        self.initDataq()
        self.lastupdate=0

    def initDataq(self):
        new_streamlist = self.props.get('datastreams',[])+['Experiment.start','Experiment.end']
        if new_streamlist != self.streamlist:
            self.streamlist=new_streamlist
            print('datamgr DataSummary: subscription reset')
            self.dataq.unsubscribe()
            self.dataq.subscribe(self.streamlist)

    def recvData(self):
        DataSummary.data_update_checks += 1
        if DataSummary.data_update_checks % 1000 == 0:
            DataSummary.data_update_checks = 0
            # if not DataSummary.got_data:
            #    print_error('datamgr.py - recvData(): No data arrived within approx. 50 sec.', 'warning')
            DataSummary.got_data = False
        while self.dataq.has_new_data():
            recvmsg=self.dataq.recv()
            #print(f'recvmsg: {recvmsg}')
            #print_error('datamgr.py - recvData(): Data incomming: {0}'.format(recvmsg), 'weak')
            A=None
            if len(recvmsg)==2:
                msg,dic=recvmsg
            elif len(recvmsg)==3:
                msg,dic,A=recvmsg
            else:
                print('datamgr message error')
                return
            DataSummary.got_data = True
            #print(f'msg: {msg}, dic: {dic}')
            self.processIncomingData(msg,dic,A)

    def processIncomingData(self,msg,newdata,A):
        '''
        '''

        incomp=self.incompleteData  # Just use the original variable self.inc...
        #print(f'newdata: {newdata}')
        if msg=='Experiment.start':
            self.expIsRunning=True
            if newdata['_task']>self.task:
                self.task=newdata['_task']
                #self.newTask()
            self.incompleteData.append(newdata)
            #print(f'incomp: {incomp}')
            #print(repr(newdata))
            self.incompleteData[-1]['_streams']=[msg]
        elif len(self.incompleteData)>0:    #process data only after experiment start has been detected
            if msg=='Experiment.end':
                self.incompleteData[-1].update(newdata)
                self.expIsRunning=False
                self.incompleteData[-1]['_streams'].append(msg)
            else:
                for line in reversed(self.incompleteData):
                    #print(line['_starttime'],newdata['timestamp'])                 #take last run starting before image was recorded
                    if line['_starttime']<newdata['timestamp']:                 #take last run starting before image was recorded
                        dic={msg+'.'+k:v for k,v in newdata.items() if k[0]!='_'}   #don't take parameters starting with _,prepend channelname to key
                        line.update(dic)
                        line['_streams'].append(msg)
                        #print()
                        #print('break')
                        #print()
                        break
                else:
                    print('datamgr.py Warning data discarded because: data arrived after timeout or more than once per run',msg)#,'incomp', repr(incomp))
        #print()
        #print('_processCompleteData()')
        #print()
        self._processCompleteData()

    def _processCompleteData(self):
        # sort out complete data
        incomp=self.incompleteData
        #print(f'incomp _pocessCompleteData: {incomp}') # this is empty at first when the ablation mirror script sends data but after a couple of ablation mirror executions it one for a sudden containes something
        if len(self.incompleteData)==0:
            return
        #print('dsdad',set(self.streamlist),set(incomp[0]['_streams']))
        while len(self.incompleteData)>0 and  set(self.streamlist)==set(self.incompleteData[0]['_streams']):
            self.completeData.append(self.incompleteData.pop(0))
            self.lastupdate=time.time()
            print('complete!!!!!!!!!!!!',self.completeData[-1])
        while len(self.incompleteData) >0 and ('_endtime' in self.incompleteData[0]
                                  and time.time() > self.incompleteData[0]['_endtime']+self.props.get('timeout', 1))\
                or self.lastupdate + self.props.get('timeout_without_endtime', 60) < time.time():
            print_error('datamgr.py - _processCompleteData(): Some data has not arrived.', 'warning')
            self.completeData.append(self.incompleteData.pop(0))
            self.lastupdate = time.time()


    def newTask(self):
        #print(self.completeData)
        pass

    def savetoFile(self, filename, folder='folder'):
        data = self.completeData
        error = False
        try:
            data_pandas = pd.DataFrame(data)

            streams_columns = [col for col in data_pandas.columns if col.startswith('_streams')]
            data_pandas.drop(columns=streams_columns, inplace=True)

            dtype_columns = [col for col in data_pandas.columns if col.endswith('.dtype')]
            data_pandas.drop(columns=dtype_columns, inplace=True)

            shape_columns = [col for col in data_pandas.columns if col.endswith('.shape')]
            data_pandas.drop(columns=shape_columns, inplace=True)

            data_pandas = data_pandas.to_records()

            # convert objects to strings
            dt_list = [dt if dt != '|O' else 'S400' for dt in np.array(data_pandas.dtype.descr)[:, 1]]
            dt_list_names = [(name, dt) for name, dt in zip(data_pandas.dtype.names, dt_list)]

            data_pandas = data_pandas.astype(dt_list_names)

            with h5py.File(filename, 'a') as f:
                f.create_dataset(folder + '/' + self.name + '%i' % time.time(), data=data_pandas)
                print_error('savetoFile(): Saving completed!', 'success')
            return True

        except Exception as e:
            print_error('savetoFile(): pandas saving failed: {0}'.format(e), 'error')
            error = True
            print_error('savetoFile(): Trying with old routine now...', 'info')

        if len(data) == 0:
            print_error('datamgr.py - savetoFile(): No data to be stored.', 'warning')
            return True

        if error:
            tabletype, flist = self._determineFormatstring(data)
            with (h5py.File(filename.replace('.h5', '_old_routine.h5'), 'a') as file):
                try:
                    dataset = file.create_dataset(folder + '/' + self.name + '%i' % time.time(), (len(data), ),
                                                  tabletype)
                except Exception as e:
                    print_error('datamgr.py - savetoFile(): Error while creating dataset with len(data)={0}'
                                ',\n{1},\ntabletype={2}'.format(len(data), e, tabletype), 'error')
                    return False
                try:
                    index = 0
                    progress = 0
                    start = datetime.datetime.now()
                    last_output = start
                    for enum, values in enumerate(data):
                        for k, v in values.items():
                            try:
                                if '_streams' not in k:
                                    # store all individual values
                                    if type(v) is list:
                                        if k != '_expName':  # BaLi's little helper wrongly emits lists
                                            for i, v_i in enumerate(v):
                                                if "{}{}".format(k,i) not in np.array(flist)[:, 0]:
                                                    print_error('datamgr.py - savetoFile(): Can\'t save k={0}, i={1},'
                                                                ' v_i={2}, v={3}, the column doesn\'t exist. Continuing'
                                                                ' to the next entry...'.format(k, i, v_i, v),
                                                                'warning')
                                                    continue
                                                if type(v_i) == str:
                                                    dataset["{}{}".format(k, i), index] = v_i.encode("ascii",
                                                                                                     "ignore")
                                                else:
                                                    dataset["{}{}".format(k, i), index] = v_i
                                    elif type(v) == str:
                                        dataset[k, index] = v.encode("ascii", "ignore")
                                    else:
                                        dataset[k, index] = v
                                # unpack lists, they should usually be short
                            except Exception as e:
                                print_error('datamgr.py - savetoFile(): Cannot store k={0}, v={1};'
                                            ' {2}.'.format(k, v, e), 'error')
                        index += 1
                        if progress + 0.1 < (enum + 1) / len(data) \
                           or (datetime.datetime.now() - last_output).seconds / 60. > 5:
                            # Every 10% progress or 5 min:
                            progress = (enum + 1) / len(data)
                            last_output = datetime.datetime.now()
                            end = start + (datetime.datetime.now() - start) / (enum + 1) * len(data)
                            print_error('datamgr.py - savetoFile(): {0} %,\tshould be done at'
                                        ' {1}.'.format(np.round(100 * progress, 1), end), 'info')
                    return True
                except Exception as e:
                    print_error('datamgr.py - savetoFile(): Error while saving into h5 '
                                'file,\n{0}'.format(e), 'error')
                    return False

    def loadFromFile(self, filepath, name, summary):
        data = []
        with h5py.File(filepath) as f:
            data = pd.DataFrame(f[name][summary][()])
            # TODO: strings.decode("utf-8")
        self.completeData = data.to_dict('records')

    def _determineFormatstring(self,data):
        keylist={}
        for run in data:
            for k, v in run.items():
                if '_streams' not in k:
                    if type(v) is list:
                        if k in keylist:
                            try:
                                v.extend(keylist[k])
                                run[k] = list(set(v))
                            except Exception as e:
                                print_error('datamgr.py - _determineFormatstring(): Error {0}'.format(e), 'error')
                                print_error('\n\nkeylist[k] = {0}\n\nv = {1}'.format(keylist[k], v), 'weak')
                                print_error('\n\nlist(set(keylist[k] + v)) = {0}'.format(list(set(keylist[k] + v))), 'weak')
            keylist.update(run)
        #print('\n\nkeylist:', keylist)
        # unpack all float and int value types
        flist = []

        # unpack all lists into numbered individual values, they should be
        # short as the head is not used to send data
        for k, v, in sorted(keylist.items()):
            if type(v) is list:
                for i, val_i in enumerate(v):
                    key_unpacked = "{}{}".format(k,i)
                    if type(val_i) == str:
                        flist.append((key_unpacked, 'S{0}'.format(len(val_i))))
                    #elif type(val_i) == bool:
                    #    flist.append((key_unpacked, int))
                    else:
                        flist.append((key_unpacked, type(val_i)))
            elif type(v) == str:
                flist.append((k, 'S{0}'.format(len(v))))
            #elif type(v) == bool:
            #    flist.append((k, int))
            else:
                flist.append((k, type(v)))
        flist = self.get_unique_flist(flist)
        tabletype = np.dtype(flist)
        return tabletype, flist

    def get_unique_flist(self, flist):
        flist = list(set(flist))
        if len(list(set(np.array(flist)[:, 0]))) == len(flist):
            return flist

        fset, counts = np.unique(np.array(flist)[:, 0], return_counts=True)

        types = {}
        for col in fset[counts > 1]:
            dt = [elt[1] for elt in flist if elt[0] == col]

        print_error('datamgr.py - get_unique_flist(): Not storing unambiguous columns {0}.'.format(types), 'warning')

        fset = fset[counts == 1]
        print(flist)
        return list(fset)

    def clear(self):
        self.incompleteData=[]
        self.lastupdate=time.time()
        self.completeData=[]
        self.task=-1

    def run_forever(self):
        while True:
            time.sleep(0.05)
            self.recvData()


class SingleDataSummary(DataSummary):
    def __init__(self,datastream,name,props=None):
        self.datastream=datastream
        super().__init__(name,props)
        self.arrays=[]
        self.streamlist=[]
        # print("SINGLE DATA SUMMARY INITIALIZED")


    def initDataq(self):
        self.dataq=DataClient(self.name)
        if self.streamlist != self.datastream:
            print('datamgr SingleDataSummary: subscription reset')
            self.dataq.unsubscribe()
            self.streamlist=self.datastream
            self.dataq.subscribe(self.streamlist)

    def processIncomingData(self,msg,newdata,A):
        self.completeData.append(newdata)
        self.arrays.append(A)
        self.lastupdate=time.time()

    def savetoFile(self,filename,folder='folder'):
        dat=self._check_datablocks()
        print('Single data Summary',self.name)
        try:
            with h5py.File(filename,'a') as file:
                dataset=file.create_dataset(folder+'/'+self.name+'dat%i'%time.time(),data=dat)
            success = super().savetoFile(filename,folder)
            return success
        except Exception as e:
            print_error('datamgr.py - SingleDataSummary - savetoFile(): Error while saving into h5 file,\n{0}'.format(e), 'error')
            return False
        # self.clear()

    def clear(self):
        ''' clear all data'''
        super().clear()
        self.arrays=[]
        self.lastupdate=time.time()

    def _check_datablocks(self):
        shapes=np.array([a.shape for a in self.arrays])
        newshape=tuple(max(i) for i in shapes.T)
        alldata=np.zeros((len(self.arrays),*newshape))
        for n,a in enumerate(self.arrays):
            alldata[n,:a.shape[0],:a.shape[1]] = a
        return alldata


class SingleImageSummary(SingleDataSummary):
    def __init__(self, datastream, name, props=None):
        super().__init__(datastream, name, props)
        self.axis = []

        # print("SINGLE IMAGE SUMMARY INITIALIZED: {}".format(name))

    def initDataq(self):
        self.dataq=ImageClient(self.name) #this is the reason for this class
        self.dataq.unsubscribe()
        self.streamlist=self.datastream
        self.dataq.subscribe(self.streamlist)

    def savetoFile(self, filename, folder):
        with h5py.File(filename,'a') as file:
            group = file.create_group(folder+'/'+self.name+'dat%i'%time.time())
            for i, a in enumerate(self.arrays):
                # save images
                group.create_dataset("{:04d}".format(i),data=a)

                # save x and yaxis
                group.create_dataset("{:04d}x".format(i),data=self.axis[i][0])
                group.create_dataset("{:04d}y".format(i),data=self.axis[i][1])

        return DataSummary.savetoFile(self, filename, folder)
        # self.clear()

    def computeImageAxis(self, newdata):
        '''
        Calculates the corrrect image Axis from image metadata. Newdata is
        the message HEAD containing metadata.
        '''
        # get image metadata from message head
        shape = newdata['shape']
        offset  = newdata['_offset']
        res = newdata['_imgresolution']

        # calculate x and y axis
        axis = [(np.arange(0,shape[i])+offset[i])*res[i] for i in range(2)]

        return axis

    def processIncomingData(self,msg,newdata,A):
        try:
            axis = self.computeImageAxis(newdata)
            self.axis.append(axis)

        except:
            print("datamgr.py: Incomplete image metadata!")
        SingleDataSummary.processIncomingData(self, msg, newdata, A)


def main(name):
    mgr = DataSummary(name)
    mgr.run_forever()


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
#    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#        QtGui.QApplication.instance().exec_()
    main('datamgr')
