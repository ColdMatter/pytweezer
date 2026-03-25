"""
┌─────────────────────────────────────────────────────────┐
│                     SERVER PROCESS                      │
│                                                         │
│  CommandServer  (ZMQ ROUTER :5562)                      │
│       │                                                 │
│       ├── 'pause'    → expManager.pause()               │
│       ├── 'restart'  → expManager.start_queue()         │
│       └── 'terminate_all' → expManager.terminate_all()  │
└─────────────────────────────────────────────────────────┘
              ▲ tcp://192.168.1.100:5562
              │
     ZMQ DEALER (client)
              │
┌─────────────────────────┐
│     CLIENT PROCESS      │
│                         │
│  CommandClient          │
│.send('pause')  ──────►│
└─────────────────────────┘
"""
