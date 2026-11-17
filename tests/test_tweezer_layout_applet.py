"""The Tweezer Layout applet: what a tap does to the layout, and the window around it."""

import pytest

from pytweezer.experiment.layouts import Lattice, Layout, Site
from pytweezer.GUI.viewers.tweezer_layout import LayoutEditor

#: A 2x2 square of 5 µm: sites at (±2.5, ±2.5), lattice points at 2.5 + 5k.
SQUARE = Lattice(kind="rectangle", spacing_um=5.0, rows=2, columns=2)


def square_editor(**kwargs) -> LayoutEditor:
    editor = LayoutEditor(**kwargs)
    editor.apply_preset(SQUARE)
    editor.dirty = False
    return editor


def positions(editor):
    return [(site.x_um, site.y_um, site.on) for site in editor.sites]


def test_tapping_a_site_switches_it_off_then_on_and_marks_the_layout_changed():
    editor = square_editor()

    editor.tap(2.3, 2.6, pick_radius_um=0.5)
    assert editor.sites[3].on is False and editor.dirty
    editor.tap(2.3, 2.6, pick_radius_um=0.5)
    assert editor.sites[3].on is True
    assert len(editor.sites) == 4


def test_grid_tap_adds_a_site_on_the_nearest_lattice_point():
    editor = square_editor()

    editor.tap(7.3, 2.0, pick_radius_um=0.5)

    assert positions(editor)[-1] == pytest.approx((7.5, 2.5, True))
    assert len(editor.sites) == 5


def test_grid_tap_that_snaps_onto_a_site_toggles_it_instead_of_adding_a_duplicate():
    editor = square_editor(min_spacing_um=0.5)

    editor.tap(6.0, 1.5, pick_radius_um=0.1)
    editor.tap(6.0, 1.5, pick_radius_um=0.1)

    assert len(editor.sites) == 5
    assert editor.sites[-1].on is False


def test_freehand_tap_lands_where_tapped():
    editor = square_editor()
    editor.mode = "freehand"

    editor.tap(10.2, -1.7, pick_radius_um=0.5)

    assert positions(editor)[-1] == pytest.approx((10.2, -1.7, True))


def test_freehand_tap_within_min_spacing_toggles_the_nearest_site_instead_of_adding():
    editor = square_editor(min_spacing_um=3.0)
    editor.mode = "freehand"

    editor.tap(4.5, 2.5, pick_radius_um=0.5)

    assert len(editor.sites) == 4
    assert editor.sites[3].on is False


def test_grid_mode_without_a_lattice_ignores_taps():
    editor = LayoutEditor()

    editor.tap(1.0, 1.0, pick_radius_um=0.5)

    assert editor.sites == [] and not editor.dirty


def test_remove_off_sites_keeps_the_on_sites_in_order():
    editor = square_editor()
    editor.sites[1].on = False

    editor.remove_off_sites()

    assert positions(editor) == [
        (-2.5, -2.5, True),
        (-2.5, 2.5, True),
        (2.5, 2.5, True),
    ]
    assert editor.counts() == (3, 0)


def test_a_layout_round_trips_through_the_editor():
    layout = Layout(
        name="ring",
        sites=[Site(x_um=0, y_um=0), Site(x_um=4, y_um=0, on=False)],
        lattice=SQUARE,
        min_spacing_um=2.0,
    )
    editor = LayoutEditor()
    editor.load(layout)

    assert editor.name == "ring" and not editor.dirty
    saved = editor.to_layout("ring")
    assert (saved.sites, saved.lattice, saved.min_spacing_um) == (
        layout.sites,
        layout.lattice,
        2.0,
    )


def test_sites_closer_than_the_minimum_spacing_are_flagged():
    editor = square_editor(min_spacing_um=6.0)

    assert editor.close_site_indices() == {0, 1, 2, 3}
    editor.min_spacing_um = 4.0
    assert editor.close_site_indices() == set()
