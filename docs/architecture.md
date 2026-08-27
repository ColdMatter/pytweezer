# Architecture

A short orientation. For working detail, read the module docstrings in the
{doc}`api/pytweezer` reference and the design notes under `docs/notes/`.

## Processes and PCs

One **server PC** runs the shared long-lived processes — the ZMQ hubs, stream
loggers, the Analysis Manager, and Device Status. **Arbitrarily many client
PCs** each run the devices physically plugged into them, their own applets, and
a GUI.

The full-control GUI (`pytweezer-server`) runs on the server PC; client PCs run
the view-only GUI (`pytweezer-client`).

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
