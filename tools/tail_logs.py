"""Print the lab-wide log stream (every host's records), one JSON line each.

The same live feed the GUI's Streams > Logs tab shows. It holds only records
published while this runs; older ones are in each host's own ``logs/`` folder.
"""

import json
import os

os.environ["PYTWEEZER_LOG_PUBLISH"] = "0"

from pytweezer.servers.messageclient import MessageClient


def main() -> None:
    client = MessageClient("log tail")
    client.subscribe("Logs")
    print("listening", flush=True)
    while True:
        msg = client.recv()
        if msg is not None:
            print(json.dumps(msg[1]), flush=True)


if __name__ == "__main__":
    main()
