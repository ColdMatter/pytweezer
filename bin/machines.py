"""See, update, start and stop pytweezer on the lab PCs over SSH.

Machines are the ``MACHINES`` table in the configuration; with no names given a
command acts on all of them. Run ``setup MACHINE`` once per PC first: it
installs this PC's SSH key (the only step that asks for a password).
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

from pytweezer import machines

#: Reaches where SSH itself got through, so the clone can still be read.
_SSH_GOT_THROUGH = ("ok", "no answer")


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand; the exit code is 0 only if every machine's result was ok."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        selected = machines.machines(args.machine)
    except ValueError as error:
        parser.error(str(error))
    if args.command == "status":
        return _status(selected)
    if args.command == "setup":
        return _setup(selected[0])
    action = {
        "update": lambda machine: machines.update(machine, args.branch),
        "start": machines.start,
        "stop": machines.stop,
    }[args.command]
    return _report(_map(action, selected))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_: str, nargs: str | int) -> argparse.ArgumentParser:
        command = commands.add_parser(name, help=help_, description=help_)
        command.add_argument("machine", nargs=nargs, metavar="MACHINE")
        return command

    add("status", "Show each PC's clone and whether pytweezer is running", "*")
    add(
        "update",
        "Fast-forward each clone and uv sync (a clone on another branch is skipped)",
        "*",
    ).add_argument("--branch", type=_branch, help="switch to this branch and update it")
    add("start", "Start the GUI on each PC's desktop", "+")
    add("stop", "Close the GUI on each PC, then clear leftover processes", "+")
    add("setup", "Install this PC's SSH key on a PC and register its launch tasks", 1)
    return parser


def _branch(name: str) -> str:
    if not machines.valid_branch(name):
        raise argparse.ArgumentTypeError(f"invalid branch name: {name!r}")
    return name


def _map(action, selected: list[machines.Machine]) -> list:
    """``action`` applied to every machine in parallel, results in ``selected`` order."""
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        return list(pool.map(action, selected))


def _status(selected: list[machines.Machine]) -> int:
    rows = [
        [
            "machine",
            "reach",
            "branch",
            "commit",
            "behind",
            "ahead",
            "dirty",
            "pytweezer",
        ]
    ]
    all_ok = True
    for machine, (gui, git) in zip(selected, _map(_inspect, selected)):
        if gui.reach != "ok":
            all_ok = False
        if gui.reach not in _SSH_GOT_THROUGH:
            rows.append([machine.name, gui.reach, "-"])
        elif isinstance(git, machines.GitState):
            rows.append(
                [
                    machine.name,
                    gui.reach,
                    git.branch,
                    git.commit,
                    str(git.behind),
                    str(git.ahead),
                    "yes" if git.dirty else "no",
                    _pytweezer(machine, gui),
                ]
            )
        else:
            all_ok = False
            rows.append([machine.name, gui.reach, f"({git.detail})"])
    _print_table(rows)
    return 0 if all_ok else 1


def _inspect(machine: machines.Machine):
    gui = machines.gui_state(machine)
    git_readable = gui.reach in _SSH_GOT_THROUGH
    return gui, (machines.git_state(machine) if git_readable else None)


def _pytweezer(machine: machines.Machine, gui: machines.GuiState) -> str:
    if gui.reach != "ok":
        return "unknown"
    if gui.owner is None:
        return "not running" if machine.os == "windows" else "-"
    owner = gui.owner
    return f"{owner['role']} ({owner['user']}, since {owner['since']})"


def _report(results: list[machines.Result]) -> int:
    rows = [["machine", "result", "detail"]]
    rows += [
        [result.machine, "ok" if result.ok else "failed", result.detail]
        for result in results
    ]
    _print_table(rows)
    return 0 if all(result.ok for result in results) else 1


def _setup(machine: machines.Machine) -> int:
    results = [machines.install_key(machine)]
    if results[0].ok and machine.os == "windows":
        results.append(machines.register_launch_tasks(machine))
    exit_code = _report(results)
    if results[0].ok and machine.repo is None and machine.os == "windows":
        _print_clones(machine)
    return exit_code


def _print_clones(machine: machines.Machine) -> None:
    """Where pytweezer clones are under the usual places, to copy into ``repo``."""
    found = machines.run(
        machine,
        r"for /d %u in (C:\Users\*) do @for /d %d in "
        r'("%u\pytweezer" "%u\ColdMatter\pytweezer") do @if exist "%d\.git" echo %d',
    )
    clones = [
        line.strip().strip('"') for line in found.stdout.splitlines() if line.strip()
    ]
    if clones:
        print(f"\nClones on {machine.name}; put one in MACHINES as its repo:")
        print("\n".join(f"  {clone}" for clone in clones))
    else:
        print(
            f"\nNo pytweezer clone found on {machine.name}; clone it, then add it to MACHINES."
        )


def _print_table(rows: list[list[str]]) -> None:
    widths = [
        max(len(row[column]) for row in rows if len(row) > column)
        for column in range(len(rows[0]))
    ]
    for row in rows:
        cells = [cell.ljust(width) for cell, width in zip(row, widths)]
        print("  ".join(cells).rstrip())


if __name__ == "__main__":
    sys.exit(main())
