"""Kill any leftover pytweezer server/device/logger processes.

Run this before starting ``pytweezer-server``/``pytweezer-client`` if a
previous session didn't shut down cleanly (e.g. the GUI process was killed
rather than closed normally) and left child processes holding their ZMQ
ports, causing the next launch to fail with "Address already in use".
"""

import argparse
import sys

from pytweezer.servers.process_cleanup import (
    ProcessListingRefused,
    kill_stale_processes,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be killed without actually killing it",
    )
    args = parser.parse_args()

    try:
        matches = kill_stale_processes(dry_run=args.dry_run)
    except ProcessListingRefused as refusal:
        print(f"Could not list processes: {refusal}")
        sys.exit(2)
    if not matches:
        print("No stale pytweezer processes found.")
        return

    verb = "Would kill" if args.dry_run else "Killed"
    for pid, category, name, _cmdline in matches:
        print(f"{verb} pid {pid}: {category}/{name}")
    print(f"{verb} {len(matches)} process(es).")


if __name__ == "__main__":
    main()
