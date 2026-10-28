"""Summaries of a :class:`~pytweezer.experiment.storage.Measurement` shared by the GUI and notebooks."""

import numpy as np


def aggregate(x, y, series=None):
    """Mean and standard error of ``y`` at each distinct ``x``, per distinct ``series``.

    Returns ``{series value: (x values, means, standard errors, counts)}``.
    Non-finite ``y`` are ignored; a single sample has a NaN standard error.
    """
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    series = np.zeros(len(y)) if series is None else np.asarray(series)
    finite = np.isfinite(y)
    curves = {}
    for value in np.unique(series):
        in_series = (series == value) & finite
        xs = np.unique(x[in_series])
        means, sems, counts = [], [], []
        for x_value in xs:
            samples = y[in_series & (x == x_value)]
            means.append(samples.mean())
            counts.append(len(samples))
            sems.append(
                samples.std(ddof=1) / np.sqrt(len(samples))
                if len(samples) > 1
                else np.nan
            )
        curves[value.item() if hasattr(value, "item") else value] = (
            xs,
            np.array(means),
            np.array(sems),
            np.array(counts),
        )
    return curves


def describe(measurement):
    """Plain-text summary: provenance, arguments, scan, and the results recorded."""
    attrs = measurement.attrs
    lines = [
        f"Task {attrs['rid']}: {attrs['experiment']}.{attrs['class_name']}",
        f"Status: {attrs['status']}    points {attrs['n_done']}/{attrs['n_points']}",
        f"Label: {attrs.get('label') or '—'}",
        f"Started {attrs.get('t_start') or '—'}, ended {attrs.get('t_end') or '—'}",
        f"Submitted by {attrs.get('submitter') or '—'} on {attrs.get('host', '?')}",
        f"Code: {attrs.get('git_commit') or '?'}"
        + (" (uncommitted changes)" if attrs.get("git_dirty") else ""),
        "",
        "Arguments:",
    ]
    scanned = {axis.argument for axis in measurement.scan.axes}
    for name, value in measurement.arguments.items():
        shown = display_value(measurement, name, value)
        lines.append(f"  {name} = {'(scanned)' if name in scanned else shown}")
    scan = measurement.scan
    lines += [
        "",
        f"Scan: {scan.order}, {scan.repetitions} repetition(s) by {scan.repeat}",
    ]
    for axis in scan.axes:
        if axis.kind == "linear":
            lines.append(
                f"  {axis.argument}: {display_value(measurement, axis.argument, axis.start)} to "
                f"{display_value(measurement, axis.argument, axis.stop)} in {axis.n} steps"
            )
        else:
            values = ", ".join(
                display_value(measurement, axis.argument, v) for v in axis.values
            )
            lines.append(f"  {axis.argument}: {values}")
    lines += ["", "Results:"]
    for name, shape in measurement.result_shapes.items():
        unit = measurement.units.get(name, "")
        lines.append(f"  {name}  {shape or 'scalar'}  {unit}")
    if measurement.constants:
        lines += ["", "Constants: " + ", ".join(measurement.constants)]
    if attrs.get("error"):
        lines += ["", attrs["error"]]
    return "\n".join(lines)


def display_value(measurement, name, value):
    """``value`` of argument ``name`` in its display unit."""
    schema = measurement.argument_schema.get(name, {})
    if schema.get("kind") != "number":
        return str(value)
    unit = schema.get("unit", "")
    return f"{value / (schema.get('scale') or 1.0):g} {unit}".strip()
