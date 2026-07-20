import numpy as np
from pytweezer.servers import DataClient,ImageClient
from pytweezer.servers import *
import argparse
class DataAverage():
    """
    The data average analysis script is subscribed to a certain datastream. It waits until
    it receives n_average datasets on this screen, then averages and publishes them.

    datastreams: set the datastreams from which the average is taken
    n_average: number of datasets to average
    filter: experiment property from head after which the average is filtered

    Behaviour:
        The average will be reset whenever
            - n_average is reached
            - the data format changes
            - the filter VALUE changes

    """
    _datastreams = PropertyAttribute('datastreams',['None'])
    _n_average = PropertyAttribute('n_average', 5)
    _filter = PropertyAttribute('filter', 'None')
    _single_experiment = PropertyAttribute('_single_experiment', 'True')

    def __init__(self, name):
        self.name = name
        self._props = Properties(name)

        self.dataq = DataClient(name.split('/')[-1])
        self.dataq.subscribe(self._datastreams)
        print('data_average.py subscriptions: ',self._datastreams)

    def run(self):
        average = None
        n_average = self._n_average
        filter_val = None
        if self._single_experiment:
            while True:
                # check if there is new data in the data queue
                if self.dataq.has_new_data():
                    msg = self.dataq.recv()
                    if msg!= None:
                        msgstr, head, data = msg
                        data = np.copy(data)
                        #print(head['_imgindex'])
                    else:
                        continue
                else:
                    continue
                #print(head['_imgindex'])

                # reset the average when new experiment starts
                if head['_imgindex'] == 0:
                    average = None

                #print('averager:',self._datastreams,'image', head['_imgindex']+1, 'received')

                try:
                    if self._filter == 'None':
                        pass
                    elif self._filter not in head:
                        print(self.name,': Filter name not in head. Proceeding as if not filtered.')
                    elif self._filter in head:
                        if head[self._filter] == filter_val:
                            pass
                        else:
                            filter_val = head[self._filter]
                            raise ValueError('Filter value has changed.')
                    average += data

                except TypeError as e:
                    #print(self.name, ': ', e)
                    #print('Creating new average array.')
                    average = data

                except ValueError as e:
                    print(self.name, ': ', e)
                    print('Data format or filter value has changed, creating fresh average array.')
                    average = data


                if head['_imgindex'] == n_average-1:
                    average /= n_average
                    self.dataq.send(head, average, '')
                    self._props.set('/Analysis/Data/axcol_bg_substract/background', average.tolist())
                    average = None



        else:
            n_count = 0
            while True:
                # check if there is new data in the data queue
                if self.dataq.has_new_data():
                    msg = self.dataq.recv()
                    if msg!= None:
                        msgstr, head, data = msg
                        data = np.copy(data)
                    else:
                        continue
                else:
                    continue

                # at this point we have a data array and want to add it to an existing
                # average array. If the array does not exist yet or was published during
                # the last iteration, it is None, and the operation causes a TypeError. If
                # the shape of the array that comes from self.dataq has changed, it will cause a ValueError.
                # In both cases we create a fresh average array.
                try:
                    if self._filter == 'None':
                        pass
                    elif self._filter not in head:
                        print(self.name,': Filter name not in head. Proceeding as if not filtered.')
                    elif self._filter in head:
                        if head[self._filter] == filter_val:
                            pass
                        else:
                            filter_val = head[self._filter]
                            raise ValueError('Filter value has changed.')
                    average += data
                    n_count += 1

                except TypeError as e:
                    #print(self.name, ': ', e)
                    #print('Creating new average array.')
                    average = data
                    n_count = 0

                except ValueError as e:
                    print(self.name, ': ', e)
                    print('Data format or filter value has changed, creating fresh average array.')
                    average = data
                    n_count = 0

                # If we have summed up enough traces, publish the average and reset.
                if n_count >= n_average-1:
                    average /= n_average
                    self.dataq.send(head, average, '')
                    self._props.set('/Analysis/Data/axcol_bg_substract/background', average.tolist())
                    n_count = 0
                    average = None

def main_run(name):
    av = DataAverage(name)
    av.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name=args.name[0]
    main_run(name)
