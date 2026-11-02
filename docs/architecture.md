# Architecture

A short orientation. For working detail, read the module docstrings in the
{doc}`api/pytweezer` reference and the design notes under `docs/notes/`.

## Processes and PCs

One **server PC** runs the shared long-lived processes — the ZMQ hubs, stream
loggers, the Properties server, the Analysis Manager, Device Status and the
Experiment Manager. **Arbitrarily many client PCs** each run the devices
physically plugged into them, their own applets, and a GUI.

The server GUI (`pytweezer-server`) runs on the server PC and starts and stops
the shared processes. Client PCs run `pytweezer-client`, which has the same tabs
except that the Servers and Loggers tabs only show the processes running on the
server PC.

## CONFIG is the single source of truth

{mod}`pytweezer.configuration.config` defines what runs where, in four
sections — `CONFIG["Servers"]`, `["Devices"]`, `["Loggers"]`, `["GUI"]` — read
through {func}`pytweezer.configuration.config.get_config`. Each entry's `host`
decides which PC that process runs on, independent of where the GUI was
launched.

Filesystem-layout constants live in {mod}`pytweezer.configuration.paths`. The
**root** `configuration/` directory (not the Python package) is unrelated JSON
data.

## The launch protocol

Every managed process is started the same way: `python <script> <name>`, where
`name` is the `CONFIG` key the script looks itself up under. Devices are all
launched through the one generic server in
{mod}`pytweezer.servers.device_server`; loggers through
{mod}`pytweezer.servers.logger_server`.

## Messaging fabric

Processes communicate over ZMQ. RPC to devices uses sipyco
({mod}`pytweezer.servers.device_client`); live image and data streams use a
pub/sub layer ({mod}`pytweezer.servers.clients`), with an XSUB/XPUB proxy
({mod}`pytweezer.servers.xsub_xpub`) forwarding between publishers and
subscribers. Streaming analysis processes
({mod}`pytweezer.analysis.analysis_base`) subscribe to a stream, transform each
message, and republish the result.

## Experiments and measurement data

The Experiment Manager ({mod}`pytweezer.servers.experiment_manager`) owns the
experiment queue. Each task runs in its own worker subprocess and reserves the
devices its experiment declares, so tasks in different pipelines run side by
side unless they need the same device. Every GUI holds a live copy of the
queue and of the points measured so far through a sipyco notifier.

Each measurement is one h5 file under the manager's `data_root`. The manager
serves those files to other PCs ({mod}`pytweezer.experiment.archive`, read by
the Results tab) and copies each finished file to `copy_root` on the lab
network drive ({mod}`pytweezer.experiment.copier`). It also holds the saved
parameters ({mod}`pytweezer.experiment.saved_parameters`): named calibration
values that experiments read and optimisers propose new values for.
