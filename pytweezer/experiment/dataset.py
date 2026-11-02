"""Load a measurement by rid and get its data out for analysis in a notebook."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pytweezer.experiment.archive import fetch_measurement
from pytweezer.experiment.storage import Measurement, find_measurement, load_measurement
from pytweezer.experiment.summary import aggregate, describe, display_value


class Dataset:
    """One measurement file, read once, with helpers for extracting and plotting it.

    ``source`` is a rid or a file path; ``results`` limits which results are
    loaded (default all). A rid is looked up under ``root`` if given, and
    otherwise fetched by :func:`~pytweezer.experiment.archive.fetch_measurement`,
    which works from any PC. Any other attribute is the
    underlying :class:`~pytweezer.experiment.storage.Measurement`'s, so
    ``ds.arguments``, ``ds.results``, ``ds.constants``, ``ds.status`` work as there.

    The ``get_*`` methods give one row per point that ran, in the order run:
    the ``/points`` columns (``index``, ``repetition``, each scanned argument,
    ``t_start``, ``t_end``) then the results, all in SI.
    """

    def __init__(
        self,
        source: int | str | Path,
        results: Iterable[str] | None = None,
        root: str | Path | None = None,
    ):
        if not isinstance(source, (int, np.integer)):
            measurement = load_measurement(source, results)
        elif root is None:
            measurement = fetch_measurement(int(source), results)
        else:
            measurement = load_measurement(find_measurement(int(source), root), results)
        self.measurement: Measurement = measurement

    def __getattr__(self, name: str) -> Any:
        if name == "measurement":
            raise AttributeError(name)
        return getattr(self.measurement, name)

    def __repr__(self) -> str:
        return (
            f"<Dataset rid={self.rid} {self.attrs['class_name']} "
            f"{self.n_done}/{self.attrs['n_points']} points, {self.status}>"
        )

    @property
    def scanned(self) -> list[str]:
        """Names of the scanned arguments, outermost axis first."""
        return [axis.argument for axis in self.scan.axes]

    @property
    def git_label(self) -> str:
        """Short commit the measurement ran, ``-dirty`` if uncommitted changes, for plots."""
        commit = self.attrs.get("git_commit")
        if not commit:
            return "git unknown"
        return f"git {commit[:7]}" + ("-dirty" if self.attrs.get("git_dirty") else "")

    @property
    def scalar_results(self) -> list[str]:
        """Loaded results with one numeric value per point."""
        return [
            name
            for name in self.results
            if self.result_shapes[name] == () and self.result_kinds[name] in "biuf"
        ]

    def get_data(
        self,
        kind: Literal["dataframe", "dict", "array"] = "dataframe",
        results: Iterable[str] | None = None,
    ):
        """The data as :meth:`get_dataframe`, :meth:`get_dict` or :meth:`get_array`."""
        getters = {
            "dataframe": self.get_dataframe,
            "dict": self.get_dict,
            "array": self.get_array,
        }
        if kind not in getters:
            raise ValueError(f"kind must be one of {list(getters)}, not {kind!r}")
        return getters[kind](results)

    def get_dict(self, results: Iterable[str] | None = None) -> dict[str, Any]:
        """Every column, array results stacked as ``(n_points, *shape)``."""
        return {**self._ordered_points(), **self._results(results)}

    def get_dataframe(self, results: Iterable[str] | None = None):
        """A :class:`pandas.DataFrame` of the results with one value per point (no images)."""
        import pandas as pd

        chosen = {
            name: value
            for name, value in self._results(results).items()
            if self.result_shapes[name] == ()
        }
        return pd.DataFrame({**self._ordered_points(), **chosen})

    def get_array(self, results: Iterable[str] | None = None) -> np.recarray:
        """A structured array of the numeric results with one value per point, fields named."""
        numeric = [
            name for name in self._results(results) if name in self.scalar_results
        ]
        return self.get_dataframe(numeric).to_records(index=False)

    def summarise(self) -> None:
        """Print the provenance, arguments, scan and results of this measurement."""
        print(describe(self.measurement))

    def plot_scan(
        self,
        y: str | None = None,
        x: str | None = None,
        series: str | None = None,
        ax=None,
    ):
        """Plot the mean and standard error of result ``y`` over repetitions against ``x``.

        ``y`` may be left out when there is only one numeric scalar result.
        ``x`` defaults to the first scanned argument (the point index if none
        was scanned) and ``series`` to the second, one curve per value. Numeric
        arguments are shown in their display unit. Returns the matplotlib axes.
        """
        # pyplot is not imported at module level: every experiment worker imports this package.
        import matplotlib.pyplot as plt

        if y is None:
            if len(self.scalar_results) != 1:
                raise ValueError(
                    f"choose y from the numeric results: {self.scalar_results}"
                )
            y = self.scalar_results[0]
        elif y not in self.scalar_results:
            raise ValueError(
                f"{y!r} is not a loaded numeric scalar result: {self.scalar_results}"
            )
        if x is None:
            x = self.scanned[0] if self.scanned else "index"
        if series is None and len(self.scanned) > 1:
            series = self.scanned[1]
        if ax is None:
            ax = plt.figure().add_subplot()

        x_values = self.points[x]
        categorical = x_values.dtype.kind not in "biuf"
        if categorical:
            labels = list(dict.fromkeys(x_values.tolist()))
            x_values = np.array([labels.index(v) for v in x_values.tolist()])
            ax.set_xticks(range(len(labels)), labels)
        else:
            x_values = x_values / (self.argument_schema.get(x, {}).get("scale") or 1.0)
        series_values = None if series is None else self.points[series]

        curves = aggregate(x_values, self.results[y], series_values)
        for value, (xs, means, sems, _counts) in curves.items():
            ax.errorbar(
                xs,
                means,
                yerr=sems if np.isfinite(sems).any() else None,
                fmt="o-",
                capsize=3,
                label=None
                if series is None
                else f"{series} = {display_value(self.measurement, series, value)}",
            )

        x_unit = "" if categorical else self.argument_schema.get(x, {}).get("unit", "")
        ax.set_xlabel(f"{x} ({x_unit})" if x_unit else x)
        y_unit = self.units.get(y, "")
        ax.set_ylabel(f"{y} ({y_unit})" if y_unit else y)
        label = self.attrs.get("label")
        ax.set_title(
            f"{self.rid}: {self.attrs['class_name']}"
            + (f" ({label})" if label else ""),
            loc="left",
        )
        ax.set_title(self.git_label, loc="right", fontsize="small", color="grey")
        if series is not None:
            ax.legend()
        return ax

    def _ordered_points(self) -> dict[str, np.ndarray]:
        first = ["index", "repetition", *self.scanned, "t_start", "t_end"]
        order = [name for name in first if name in self.points]
        order += [name for name in self.points if name not in order]
        return {name: self.points[name] for name in order}

    def _results(self, names: Iterable[str] | None) -> dict[str, Any]:
        if names is None:
            return self.results
        return {name: self.results[name] for name in names}
