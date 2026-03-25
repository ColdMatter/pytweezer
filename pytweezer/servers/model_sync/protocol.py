"""
protocol.py
-----------
Defines the message format exchanged between ModelServer and ModelClient.
All messages are JSON-serialisable dicts with a mandatory 'op' field.

Operations
----------
  Client → Server  (REQ/REP)
  ┌────────────┬──────────────────────────────────────────────────┐
  │ op         │ payload fields                                   │
  ├────────────┼──────────────────────────────────────────────────┤
  │ 'INIT'     │ model_name                                       │
  │ 'SET'      │ model_name, key, value                           │
  │ 'DEL'      │ model_name, key                                  │
  └────────────┴──────────────────────────────────────────────────┘

  Server → All Clients  (PUB, topic = model_name)
  ┌────────────┬──────────────────────────────────────────────────┐
  │ op         │ payload fields                                   │
  ├────────────┼──────────────────────────────────────────────────┤
  │ 'SET'      │ model_name, key, value                           │
  │ 'DEL'      │ model_name, key                                  │
  └────────────┴──────────────────────────────────────────────────┘
"""

import json


def encode(msg: dict) -> bytes:
    return json.dumps(msg).encode('utf-8')


def decode(raw: bytes) -> dict:
    return json.loads(raw.decode('utf-8'))


# ── Outbound helpers (client → server) ──────────────────────────────────────

def make_init(model_name: str) -> dict:
    return {'op': 'INIT', 'model_name': model_name}


def make_set(model_name: str, key, value) -> dict:
    """
    Keys are always converted to strings for JSON compatibility.
    Integer keys (e.g. task numbers) are restored on decode via
    the model's key_type callable.
    """
    return {'op': 'SET', 'model_name': model_name,
            'key': str(key), 'value': value}


def make_del(model_name: str, key) -> dict:
    return {'op': 'DEL', 'model_name': model_name, 'key': str(key)}


# ── Response helpers (server → client REP) ───────────────────────────────────

def ok(payload=None) -> dict:
    return {'status': 'OK', 'payload': payload}


def err(reason: str) -> dict:
    return {'status': 'ERR', 'reason': reason}