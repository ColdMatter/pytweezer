from pytweezer.servers import DataClient,ImageClient
from pytweezer.servers import Properties,PropertyAttribute
import argparse

class ExperimentTimer():

    def __init__(self,name):
        self._props = Properties(name)
        self.dataqstart = DataClient(name.split('/')[-1])
        self.dataqend = DataClient(name.split('/')[-1])
        self.dataqstart.subscribe('Experiment.start')
        self.dataqend.subscribe('Experiment.end')
        self.name = name
        #print(name)

    def run(self):
        starttime = 0
        endtime = 0 
        while True:
            if self.dataqstart.has_new_data():
                msg = self.dataqstart.recv()
                if msg!= None:
                    msgstr, head = msg
                    starttime = head['_starttime']
            if self.dataqend.has_new_data():
                msg = self.dataqend.recv()
                if msg!= None:
                    msgstr, head = msg
                    endtime = head['_endtime']
            if starttime < endtime:
                head['duration'] = endtime - starttime 
                head['timestamp'] = endtime 
                self.dataqstart.send(head, [], '')
                starttime, endtime = 0, 0



def main_run(name):
    timer = ExperimentTimer(name)
    timer.run()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name=args.name[0]
    main_run(name)
