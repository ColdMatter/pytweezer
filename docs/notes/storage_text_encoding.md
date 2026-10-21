# Storage: one encoding for text results

Status: agreed direction, not yet implemented. Scope is
`pytweezer/experiment/storage.py`.

## The bug

A result recorded as a dict at one point and as a string at another makes the
whole measurement file unloadable:

```python
# point 0
writer.record("d", {"a": 1})
# point 1
writer.record("d", "plain")
load_measurement(path)  # JSONDecodeError, for every result in the file
```

Both values are stored as h5 strings. The dataset's `encoding` attribute is set
by whichever kind arrives first (`"json"` for a dict, `""` for a string), and
`_check_fits` only asks "is this text?", so the second value is accepted.
`_decode` then runs `json.loads` over every entry of a `"json"` column. In the
opposite order, string first and dict later, nothing fails, but the dict loads
back as raw JSON text.

Nothing goes wrong until someone loads the data, which is why this matters.

## What we want to change

Make the bad state impossible to write rather than checking for it, by
adopting one storage rule:

> Numbers and numeric arrays are stored as typed arrays. Everything else
> (strings, dicts) is stored as JSON text. A result's kind and shape are fixed
> by its first record.

- `_encode` stores strings as JSON as well (`"plain"` → `"\"plain\""`), so every
  text dataset has one encoding and decodes the same way.
- State the rule in the `record()` docstring.
- Bump `SCHEMA_VERSION` to 2. The reader keeps decoding version-1 files whose
  text datasets have `encoding == ""` as plain strings.
- Add one test in `tests/test_experiment_storage.py`, parametrised over both
  orders (dict then str, str then dict).

## Still to decide

Should a string at one point and a dict at another be **recorded as given** or
**refused** with a `RecordError`? Either way the file stays loadable. Mixing them is
probably a mistake in the experiment code; refusing catches that, recording it
is the more literal reading of "record faithfully".

## Considered and deferred

- **Check the encoding in `_check_fits` only.** This fixes the symptom but keeps
  two ways of storing text.
- **Experiments declare their results up front** (name, type, shape, unit), as
  they already declare arguments. This is the strongest form of the rule, but it
  changes how every experiment is written. Revisit once the results experiments
  produce have settled.
- **Make the reader tolerate a bad column.** This would hide corrupt data
  instead of preventing it.

## Also found, separate fix

Complex-valued results cannot be recorded: `_record_result` uses `np.nan` as the
fill value for complex dtypes, which h5py rejects. Use `complex(np.nan, np.nan)`.
It fails loudly, so it's lower priority.
