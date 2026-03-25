"""
protocol.py
-----------
Command messages are simple JSON dicts:

  Client → Server
  ┌─────────────┬────────────────────────────────────────┐
  │ Field       │ Description                            │
  ├─────────────┼────────────────────────────────────────┤
  │ cmd         │ Command name string, e.g. 'pause'      │
  │ args        │ Optional dict of arguments             │
  │ msg_id      │ UUID string for reply matching         │
  └─────────────┴────────────────────────────────────────┘

  Server → Client  (reply)
  ┌─────────────┬────────────────────────────────────────┐
  │ Field       │ Description                            │
  ├─────────────┼────────────────────────────────────────┤
  │ status      │ 'OK' or 'ERR'                          │
  │ msg_id      │ Echoed from the request                │
  │ payload     │ Optional return data                   │
  │ reason      │ Error description (ERR only)           │
  └─────────────┴────────────────────────────────────────┘
"""

import json
import uuid


def encode(msg: dict) -> bytes:
    return json.dumps(msg).encode('utf-8')


def decode(raw: bytes) -> dict:
    return json.loads(raw.decode('utf-8'))


def make_command(cmd: str, args: dict = None) -> dict:
    return {
        'cmd'   : cmd,
        'args'  : args or {},
        'msg_id': str(uuid.uuid4()),
    }


def ok(msg_id: str, payload=None) -> dict:
    return {'status': 'OK', 'msg_id': msg_id, 'payload': payload}


def err(msg_id: str, reason: str) -> dict:
    return {'status': 'ERR', 'msg_id': msg_id, 'reason': reason}