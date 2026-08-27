# pytweezer

Control software for an atom-tweezer experiment (Rb and CaF systems). PyQt
GUIs, ZMQ messaging, and sipyco RPC servers coordinate device drivers
(cameras, MotMaster experiment sequencers) across the lab PCs.

One **server PC** runs the shared long-lived processes; **arbitrarily many
client PCs** each run the devices plugged into them plus their own applets and
GUI. {doc}`architecture` sketches how the pieces fit together, and the
{doc}`api/pytweezer` reference is generated from the package docstrings.

```{toctree}
:maxdepth: 2
:caption: Contents

getting-started
architecture
api/pytweezer
```

```{note}
This site is a skeleton. The narrative pages are deliberately short — extend
them as the need arises. Longer-form design notes that are not part of this
site live in `docs/notes/`.
```
