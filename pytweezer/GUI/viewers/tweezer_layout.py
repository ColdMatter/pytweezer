"""Draw a tweezer array on the focal plane and save it as a named layout.

Sites are placed by tapping: in Grid mode a tap snaps to the lattice of the
last preset, in Freehand mode it lands where tapped. Tapping a site switches it
on or off. Layouts are saved to the Experiment Manager (see
:mod:`pytweezer.experiment.layouts`), and "Put on SLM" queues
``SlmTweezerLayout`` with the open layout.
"""

from typing import Literal

import numpy as np

from pytweezer.experiment.layouts import (
    Lattice,
    Layout,
    Site,
    close_pairs,
    nearest_lattice_point,
    preset_sites,
)

#: Distance (µm) within which a snapped lattice point counts as an existing site.
_SAME_POINT_UM = 1e-6


class LayoutEditor:
    """The layout being edited, and what a tap does to it. No Qt."""

    def __init__(self, min_spacing_um: float = 3.0) -> None:
        self.sites: list[Site] = []
        self.lattice: Lattice | None = None
        self.mode: Literal["grid", "freehand"] = "grid"
        self.min_spacing_um = float(min_spacing_um)
        self.name: str | None = None
        self.dirty = False

    def _nearest(self, x_um: float, y_um: float) -> tuple[int | None, float]:
        if not self.sites:
            return None, np.inf
        distances = [np.hypot(s.x_um - x_um, s.y_um - y_um) for s in self.sites]
        index = int(np.argmin(distances))
        return index, float(distances[index])

    def _toggle(self, index: int) -> None:
        self.sites[index].on = not self.sites[index].on
        self.dirty = True

    def _add(self, x_um: float, y_um: float) -> None:
        self.sites.append(Site(x_um=x_um, y_um=y_um))
        self.dirty = True

    def tap(self, x_um: float, y_um: float, pick_radius_um: float) -> None:
        """Toggle the site under the tap, or add one there.

        A tap within ``pick_radius_um`` (or half the minimum spacing, if more)
        of a site toggles it. Otherwise Grid mode adds a site at the nearest
        lattice point, toggling the site already there if there is one, and
        Freehand mode adds one at the tap unless a site is nearer than the
        minimum spacing, which then toggles instead.
        """
        index, distance = self._nearest(x_um, y_um)
        if distance <= max(pick_radius_um, self.min_spacing_um / 2):
            self._toggle(index)
        elif self.mode == "grid":
            if self.lattice is None:
                return
            point = nearest_lattice_point(self.lattice, x_um, y_um)
            index, distance = self._nearest(*point)
            if distance <= _SAME_POINT_UM:
                self._toggle(index)
            else:
                self._add(*point)
        elif distance < self.min_spacing_um:
            self._toggle(index)
        else:
            self._add(x_um, y_um)

    def apply_preset(self, lattice: Lattice) -> None:
        self.sites = preset_sites(lattice)
        self.lattice = lattice
        self.dirty = True

    def remove_off_sites(self) -> None:
        kept = [site for site in self.sites if site.on]
        if len(kept) != len(self.sites):
            self.sites = kept
            self.dirty = True

    def load(self, layout: Layout) -> None:
        self.sites = [site.model_copy() for site in layout.sites]
        self.lattice = layout.lattice
        self.min_spacing_um = layout.min_spacing_um
        self.name = layout.name
        self.dirty = False

    def to_layout(self, name: str) -> Layout:
        return Layout(
            name=name,
            sites=[site.model_copy() for site in self.sites],
            lattice=self.lattice,
            min_spacing_um=self.min_spacing_um,
        )

    def close_site_indices(self) -> set[int]:
        return {
            i for pair in close_pairs(self.sites, self.min_spacing_um) for i in pair
        }

    def counts(self) -> tuple[int, int]:
        on = sum(site.on for site in self.sites)
        return on, len(self.sites) - on
