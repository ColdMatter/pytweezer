"""Tweezer layouts: named sets of trap sites, shared through the Experiment Manager.

A :class:`Layout` is a list of :class:`Site` positions in µm in the focal plane,
relative to the blaze spot (the frame the SLM hologram code uses), each on or
off. Only the on sites are trapped; off sites are kept so they can be switched
back on. A layout may carry the :class:`Lattice` it was drawn on, so the editor
can snap new sites to it.

:class:`LayoutBook` holds the rules for saving, finding and deleting layouts;
:class:`LayoutStore` persists the book as JSON. Pure numpy: importable without
the GPU stack.
"""

from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator
from scipy.spatial import cKDTree

from pytweezer.experiment.queue import ModelStore, now


class LayoutError(ValueError):
    """A layout request that can't be carried out."""


class Site(BaseModel):
    x_um: float
    y_um: float
    on: bool = True


class Lattice(BaseModel):
    """A preset's lattice. ``rows``/``columns`` are for rectangles, ``sites`` for kagome."""

    kind: Literal["rectangle", "kagome"]
    spacing_um: float
    angle_deg: float = 0.0
    rows: int | None = None
    columns: int | None = None
    sites: int | None = None


class Layout(BaseModel):
    """A named tweezer layout; ``submitter`` is whoever saved it."""

    name: str
    sites: list[Site]
    lattice: Lattice | None = None
    min_spacing_um: float = 3.0
    submitter: str = ""
    saved_at: datetime = Field(default_factory=now)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, name: str) -> str:
        name = name.strip()
        if not name:
            raise ValueError("a layout needs a name")
        return name

    @property
    def on_sites(self) -> tuple[np.ndarray, np.ndarray]:
        """``x_um`` and ``y_um`` of the on sites, in list order."""
        on = [site for site in self.sites if site.on]
        return (
            np.array([site.x_um for site in on], dtype=float),
            np.array([site.y_um for site in on], dtype=float),
        )


def _rotate(x: np.ndarray, y: np.ndarray, angle_deg: float):
    angle = np.radians(angle_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    return x * cos - y * sin, x * sin + y * cos


def _sites(x: np.ndarray, y: np.ndarray) -> list[Site]:
    return [Site(x_um=float(xi), y_um=float(yi)) for xi, yi in zip(x, y)]


def rectangle(
    rows: int, columns: int, spacing_um: float, angle_deg: float = 0.0
) -> list[Site]:
    """A centred ``rows`` x ``columns`` array, row by row, rotated anticlockwise.

    The same positions, in the same order, as the phase-mask generator's
    ``generate_weighted_array``.
    """
    x = (np.arange(columns) - (columns - 1) / 2) * spacing_um
    y = (np.arange(rows) - (rows - 1) / 2) * spacing_um
    grid_x, grid_y = np.meshgrid(x, y)
    return _sites(*_rotate(grid_x.ravel(), grid_y.ravel(), angle_deg))


#: Kagome lattice in units of the nearest-neighbour spacing: the centre
#: lattice vectors (columns) and the three sites about each centre.
_KAGOME_VECTORS = np.array([[2.0, 1.0], [0.0, np.sqrt(3)]])
_KAGOME_BASIS = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2], [-0.5, np.sqrt(3) / 2]])


def kagome(sites: int, spacing_um: float, angle_deg: float = 0.0) -> list[Site]:
    """The closed shell of a kagome lattice whose site count is nearest ``sites``."""
    k_max = int(1.5 * np.sqrt(sites)) + 2
    n, m = (index.ravel() for index in np.mgrid[-k_max : k_max + 1, -k_max : k_max + 1])
    centres = np.column_stack([m, n]) @ _KAGOME_VECTORS.T * spacing_um
    every_site = np.vstack([centres + b * spacing_um for b in _KAGOME_BASIS])
    radii = np.round(np.linalg.norm(every_site, axis=1), decimals=5)
    shells, counts = np.unique(radii, return_counts=True)
    radius = shells[np.argmin(np.abs(np.cumsum(counts) - sites))]
    kept = every_site[radii <= radius]
    return _sites(*_rotate(kept[:, 0], kept[:, 1], angle_deg))


def preset_sites(lattice: Lattice) -> list[Site]:
    if lattice.kind == "rectangle":
        return rectangle(
            lattice.rows, lattice.columns, lattice.spacing_um, lattice.angle_deg
        )
    return kagome(lattice.sites, lattice.spacing_um, lattice.angle_deg)


def _unit_cell(lattice: Lattice) -> tuple[np.ndarray, np.ndarray]:
    """Lattice vectors (as columns) and basis offsets, in µm, before rotation."""
    s = lattice.spacing_um
    if lattice.kind == "kagome":
        return _KAGOME_VECTORS * s, _KAGOME_BASIS * s
    # A rectangle with an even number of columns (rows) has no site on the
    # centre line, so its lattice is offset by half a spacing.
    offset = [0.5 * s * (n % 2 == 0) for n in (lattice.columns or 1, lattice.rows or 1)]
    return np.eye(2) * s, np.array([offset])


def lattice_points(
    lattice: Lattice,
    x_range_um: tuple[float, float],
    y_range_um: tuple[float, float],
) -> np.ndarray:
    """Every point of ``lattice`` in the window, as ``(n, 2)`` rows of ``x_um, y_um``."""
    vectors, basis = _unit_cell(lattice)
    corners = np.array([[x, y] for x in x_range_um for y in y_range_um], dtype=float)
    unrotated = np.column_stack(
        _rotate(corners[:, 0], corners[:, 1], -lattice.angle_deg)
    )
    points = []
    for offset in basis:
        indices = np.linalg.solve(vectors, (unrotated - offset).T)
        low = np.floor(indices.min(axis=1)).astype(int) - 1
        high = np.ceil(indices.max(axis=1)).astype(int) + 1
        i, j = np.mgrid[low[0] : high[0] + 1, low[1] : high[1] + 1]
        points.append(np.column_stack([i.ravel(), j.ravel()]) @ vectors.T + offset)
    points = np.vstack(points)
    x, y = _rotate(points[:, 0], points[:, 1], lattice.angle_deg)
    inside = (
        (x >= x_range_um[0])
        & (x <= x_range_um[1])
        & (y >= y_range_um[0])
        & (y <= y_range_um[1])
    )
    return np.column_stack([x[inside], y[inside]])


def nearest_lattice_point(
    lattice: Lattice, x_um: float, y_um: float
) -> tuple[float, float]:
    reach = 2 * lattice.spacing_um
    points = lattice_points(
        lattice, (x_um - reach, x_um + reach), (y_um - reach, y_um + reach)
    )
    nearest = points[np.argmin(np.hypot(points[:, 0] - x_um, points[:, 1] - y_um))]
    return float(nearest[0]), float(nearest[1])


def close_pairs(sites: list[Site], min_spacing_um: float) -> list[tuple[int, int]]:
    """Index pairs ``(i, j)``, ``i < j``, of sites nearer than ``min_spacing_um``."""
    if len(sites) < 2:
        return []
    xy = np.array([[site.x_um, site.y_um] for site in sites])
    pairs = cKDTree(xy).query_pairs(min_spacing_um, output_type="ndarray")
    # query_pairs includes pairs exactly min_spacing_um apart.
    near = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T) < min_spacing_um
    return sorted((int(i), int(j)) for i, j in pairs[near])


class LayoutState(BaseModel):
    schema_version: int = 1
    layouts: list[Layout] = Field(default_factory=list)


class LayoutBook:
    """Every saved layout, and the rules for changing them."""

    def __init__(self, state: LayoutState | None = None) -> None:
        self.state = state or LayoutState()

    def get(self, name: str) -> Layout:
        name = name.strip()
        for layout in self.state.layouts:
            if layout.name == name:
                return layout
        raise LayoutError(f"no layout {name!r}")

    def save(self, layout: Layout, overwrite: bool = False) -> Layout:
        """Store ``layout``, replacing one of the same name only with ``overwrite``."""
        layout = layout.model_copy(update={"saved_at": now()})
        for index, existing in enumerate(self.state.layouts):
            if existing.name == layout.name:
                if not overwrite:
                    raise LayoutError(f"a layout {layout.name!r} already exists")
                self.state.layouts[index] = layout
                return layout
        self.state.layouts.append(layout)
        return layout

    def delete(self, name: str) -> None:
        self.state.layouts.remove(self.get(name))

    def find(self) -> list[Layout]:
        return sorted(self.state.layouts, key=lambda layout: layout.name)


class LayoutStore(ModelStore):
    """Persists a :class:`LayoutState`."""

    model = LayoutState
