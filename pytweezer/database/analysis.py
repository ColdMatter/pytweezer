"""Notebook helpers: read monitor readings, and line them up with scan points.

::

    from pytweezer.database.analysis import readings, points_with_readings

    readings("ni_adc", ["ai0"], start="2026-10-07 09:00")
    points_with_readings(1234, "ni_adc", ["ai0", "ai1"])  # one row per point

A point takes the ``agg`` of the readings inside its own time window. A point
shorter than the logger's interval usually has none, so it takes the last
reading before it ended instead.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pytweezer.configuration.config import DATABASE
from pytweezer.database.backfill import load_with_scalars, point_rows
from pytweezer.experiment.storage import Measurement, data_root


def _connect(dsn=None):
    import psycopg

    return psycopg.connect(dsn or DATABASE["dsn"], connect_timeout=3)


def _as_datetime(value):
    """A timezone-aware datetime; a naive one is taken as this PC's local time."""
    return pd.Timestamp(value).to_pydatetime().astimezone()


def _long_readings(measurement, fields, start, end, tags, dsn):
    sql = "SELECT time, field, value FROM readings WHERE measurement = %s"
    params = [measurement]
    if fields:
        sql += " AND field = ANY(%s)"
        params.append(list(fields))
    if start is not None:
        sql += " AND time >= %s"
        params.append(_as_datetime(start))
    if end is not None:
        sql += " AND time <= %s"
        params.append(_as_datetime(end))
    if tags:
        sql += " AND tags @> %s::jsonb"
        params.append(json.dumps({k: str(v) for k, v in tags.items()}))
    sql += " ORDER BY time"
    with _connect(dsn) as conn:
        rows = conn.execute(sql, params).fetchall()
    return pd.DataFrame(rows, columns=["time", "field", "value"])


def readings(measurement, fields=None, start=None, end=None, tags=None, dsn=None):
    """Readings of one measurement as a DataFrame indexed by time, a column per field.

    Args:
        measurement (str): e.g. ``"ni_adc"``.
        fields (list[str], optional): only these fields; default all.
        start, end: anything ``pd.Timestamp`` accepts; naive times are local.
        tags (dict, optional): only readings carrying all of these tags.
        dsn (str, optional): connection string; defaults to ``DATABASE["dsn"]``.
    """
    long = _long_readings(measurement, fields, start, end, tags, dsn)
    if long.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="time"))
    return long.pivot_table(index="time", columns="field", values="value")


def aggregate_by_point(t_start, t_end, times, values, agg="mean"):
    """Aggregate ``values`` (sorted by ``times``) over each ``[t_start, t_end]`` window.

    All times are epoch seconds. A window with no readings takes the last
    reading before its end, or NaN if there is none.
    """
    reduce = getattr(np, agg) if isinstance(agg, str) else agg
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    lo = np.searchsorted(times, t_start, side="left")
    hi = np.searchsorted(times, t_end, side="right")
    out = np.full(len(lo), np.nan)
    for i, (a, b) in enumerate(zip(lo, hi, strict=True)):
        if b > a:
            out[i] = reduce(values[a:b])
        elif b > 0:
            out[i] = values[b - 1]
    return out


def _measurement_for(source, dsn):
    if isinstance(source, Measurement):
        return source
    if isinstance(source, int | np.integer):
        with _connect(dsn) as conn:
            row = conn.execute(
                "SELECT h5_path FROM runs WHERE rid = %s", (int(source),)
            ).fetchone()
        if row is None or row[0] is None:
            raise LookupError(f"no measurement file recorded for rid {source}")
        source = data_root() / row[0]
    return load_with_scalars(Path(source))


def points_with_readings(
    source,
    measurement,
    fields=None,
    agg="mean",
    tags=None,
    lookback_s=600.0,
    dsn=None,
):
    """One row per scan point: its scanned values, scalar results and readings.

    Args:
        source: an rid (its file is looked up in the ``runs`` table), an h5
            path, or a loaded :class:`~pytweezer.experiment.storage.Measurement`.
        measurement (str): the readings to attach, e.g. ``"ni_adc"``.
        fields (list[str], optional): which fields; default all.
        agg: ``"mean"``, ``"median"``, ``"min"``, ``"max"``, ``"std"`` or a
            callable reducing an array to a number.
        tags (dict, optional): only readings carrying all of these tags.
        lookback_s (float): how far before the run to look for the reading a
            short first point falls back on.

    Readings columns are named ``"<measurement>/<field>"``.
    """
    run = _measurement_for(source, dsn)
    rows = point_rows(run)
    table = pd.DataFrame(
        [
            {
                "point_index": row["point_index"],
                "t_start": row["t_start"],
                "t_end": row["t_end"],
                **row["scan_values"],
                **row["scalars"],
            }
            for row in rows
        ]
    )
    if table.empty:
        return table
    start = datetime.fromtimestamp(table["t_start"].min() - lookback_s, UTC)
    end = datetime.fromtimestamp(table["t_end"].max(), UTC)
    long = _long_readings(measurement, fields, start, end, tags, dsn)
    for field, series in long.groupby("field", sort=True):
        times = series["time"].map(lambda t: t.timestamp()).to_numpy()
        table[f"{measurement}/{field}"] = aggregate_by_point(
            table["t_start"], table["t_end"], times, series["value"], agg
        )
    for column in ("t_start", "t_end"):
        table[column] = pd.to_datetime(table[column], unit="s", utc=True)
    return table.set_index("point_index")
