"""
┌─────────────────────────────────────────────────────┐
│                    SERVER PROCESS                   │
│                                                     │
│  ScheduleModel (authoritative)                      │
│       │  mutate                                     │
│       ▼                                             │
│  ModelServer (ZMQ REP + PUB)                        │
│    REP :5560  ◄── handles commands (set/del/init)   │
│    PUB :5561  ──► broadcasts changes to all clients │
└─────────────────────────────────────────────────────┘
          │ TCP                        │ TCP
          ▼                            ▼
┌──────────────────┐        ┌──────────────────┐
│  REMOTE CLIENT A │        │  REMOTE CLIENT B │
│                  │        │                  │
│  ScheduleModel   │        │  ScheduleModel   │
│  (replica)       │        │  (replica)       │
│  ModelClient     │        │  ModelClient     │
│  SUB :5561       │        │  SUB :5561       │
│  REQ :5560       │        │  REQ :5560       │
└──────────────────┘        └──────────────────┘
"""


"""
# ── On the SERVER machine ────────────────────────────────────────────────────

from pytweezer.servers.model_sync.server import ModelServer
from pytweezer.models.schedule_model import ScheduleModel

app = QApplication(sys.argv)

schedule = ScheduleModel({})   # authoritative model — attach to your QTableView

server = ModelServer(
    rep_endpoint='tcp://*:5560',
    pub_endpoint='tcp://*:5561',
)
server.register('schedule', schedule, key_type=int)
server.start()


# ── On a REMOTE CLIENT machine ───────────────────────────────────────────────

from pytweezer.servers.model_sync.client import ModelClient
from pytweezer.models.schedule_model import ScheduleModel

app = QApplication(sys.argv)

replica = ScheduleModel({})    # starts empty — filled by INIT snapshot

client = ModelClient(
    model       = replica,
    model_name  = 'schedule',
    key_type    = int,
    server_ip   = '192.168.1.100',   # server machine IP
    rep_port    = 5560,
    pub_port    = 5561,
)
client.start()

# Mutate via client — propagates to server and all other replicas
client.set(42, {
    'task': 42, 'label': 'Run A', 'expName': 'exp1',
    'status': 'pending', 'repetition': 1,
    'run': 0, 'priority': 5, 'dueDateTime': '...'
})
client.delete(42)
"""
