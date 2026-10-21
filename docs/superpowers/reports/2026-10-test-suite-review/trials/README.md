# Trial runs (before the design)

Agents wrote tests for `experiment/scan.py`, `experiment/storage.py` and
`drivers/slm.py` twice:

- `baseline/`: written without the `write-tests` skill
- `green/`: written with it

Neither run was blind: the `green/` agents read the existing test files. These
are for stage 2 only, as a second opinion, and stage 1 never saw them.

The `green/` storage run found two bugs, recorded in
`docs/notes/storage_text_encoding.md` on the `storage-text-encoding` branch.
The `green/` SLM run confirmed its tests catch real faults: when it broke the
driver's data conversion, 6 of its 10 tests failed.

The files have a `.py.txt` suffix so that pytest and ruff don't pick them up.
