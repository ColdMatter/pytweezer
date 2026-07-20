from pytweezer.servers import Properties,DataClient,PropertyAttribute
import time


class DataStreamLogger():
    """ Provide information about active streams in Property tree

    Each stream that sends data is appended to a dictionary called 'active'
    A timestamp for the last activity is recorded
    Streams that have been inactive for longer than maxStreamInactive are removed 

    """
    active=PropertyAttribute('active',{})    
    maxage=PropertyAttribute('maxStreamInactive[s]',1000)

    def __init__(self,name):
        self._props=Properties(name)
        self.stream=DataClient(name)
        self.stream.subscribe('')       #listen to all streams
        self.updated = False 

    def run(self):
        ''' detect active channels'''
        lastupdate=time.time()
        active=self.active
        while True:
            msg=self.stream.recv()
            self.updated=False
            if msg != None:
                active[msg[0]]={'timestamp':time.time(),'parts':len(msg)}
                
                self.updated=True
            if (time.time()-lastupdate>1): #update dictionary every second only
                active=self.removeInactiveStreams(active)
                if self.updated:
                    self.active=active
                    updated=False
                    lastupdate=time.time()

    def removeInactiveStreams(self,streamlist):
        """ remove streams older than maxage from list 

        :streamlist : dictionary of streams to be filtered
        :maxage :   maximum age of stream in seconds
        """
        new= {k:v for k,v in streamlist.items() if time.time()-v['timestamp']<self.maxage}
        if new!=streamlist:
            self.updated=True
        return new

def run_DataLogger(name):
    d=DataStreamLogger(name)
    d.run()


if __name__ == "__main__":
    name='Servers/DataStream'
    run_DataLogger(name)

