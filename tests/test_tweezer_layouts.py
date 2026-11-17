import numpy as np
import pytest
from pydantic import ValidationError

from pytweezer.experiment.layouts import (
    Lattice,
    Layout,
    LayoutBook,
    LayoutError,
    LayoutState,
    LayoutStore,
    Site,
    close_pairs,
    kagome,
    lattice_points,
    nearest_lattice_point,
    preset_sites,
    rectangle,
)


def test_rectangle_matches_the_generators_positions_and_order():
    sites = rectangle(2, 3, 5.0, angle_deg=2.5)

    # As generate_weighted_array: centred, row-major with y slowest, rotated anticlockwise
    X, Y = np.meshgrid(np.linspace(-5, 5, 3), np.linspace(-2.5, 2.5, 2))
    a = np.radians(2.5)
    expected_x = X.ravel() * np.cos(a) - Y.ravel() * np.sin(a)
    expected_y = X.ravel() * np.sin(a) + Y.ravel() * np.cos(a)
    np.testing.assert_allclose([s.x_um for s in sites], expected_x)
    np.testing.assert_allclose([s.y_um for s in sites], expected_y)
    assert all(s.on for s in sites)


def test_kagome_neighbours_are_one_spacing_apart_and_on_the_lattice():
    sites = kagome(30, 4.0, angle_deg=10.0)

    xy = np.array([[s.x_um, s.y_um] for s in sites])
    distances = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    np.fill_diagonal(distances, np.inf)
    np.testing.assert_allclose(distances.min(axis=1), 4.0)
    lattice = Lattice(kind="kagome", spacing_um=4.0, angle_deg=10.0)
    for x, y in xy:
        assert nearest_lattice_point(lattice, x, y) == pytest.approx((x, y))


@pytest.mark.parametrize("rows, columns", [(2, 3), (3, 3), (4, 2)])
def test_a_tap_near_a_rectangle_site_snaps_onto_it(rows, columns):
    lattice = Lattice(
        kind="rectangle", spacing_um=5.0, angle_deg=3.0, rows=rows, columns=columns
    )
    for s in preset_sites(lattice):
        snapped = nearest_lattice_point(lattice, s.x_um + 1.0, s.y_um - 1.0)
        assert snapped == pytest.approx((s.x_um, s.y_um))


def test_lattice_points_cover_the_window_and_nothing_outside():
    lattice = Lattice(kind="rectangle", spacing_um=5.0, rows=3, columns=3)

    points = lattice_points(lattice, (-11, 11), (-6, 6))

    assert len(points) == 5 * 3
    assert np.all(np.abs(points[:, 0]) <= 11) and np.all(np.abs(points[:, 1]) <= 6)


def test_close_pairs_finds_only_sites_nearer_than_the_minimum():
    sites = [
        Site(x_um=0, y_um=0),
        Site(x_um=2, y_um=0, on=False),
        Site(x_um=10, y_um=0),
    ]
    assert close_pairs(sites, 3.0) == [(0, 1)]


def test_on_sites_keep_list_order_and_leave_out_off_sites():
    layout = Layout(
        name="three",
        sites=[
            Site(x_um=3, y_um=1),
            Site(x_um=2, y_um=2, on=False),
            Site(x_um=1, y_um=3),
        ],
    )
    x_um, y_um = layout.on_sites
    np.testing.assert_array_equal(x_um, [3, 1])
    np.testing.assert_array_equal(y_um, [1, 3])


def test_a_name_is_stripped_and_cannot_be_blank():
    assert Layout(name="  ring ", sites=[]).name == "ring"
    with pytest.raises(ValidationError):
        Layout(name="  ", sites=[])


def test_book_overwrites_only_when_asked():
    book = LayoutBook()
    book.save(Layout(name="ring", sites=[Site(x_um=0, y_um=0)]))

    with pytest.raises(LayoutError, match="already exists"):
        book.save(Layout(name="ring", sites=[Site(x_um=1, y_um=0)]))
    book.save(Layout(name="ring", sites=[Site(x_um=1, y_um=0)]), overwrite=True)
    book.save(Layout(name="arc", sites=[Site(x_um=0, y_um=0)]))

    assert [layout.name for layout in book.find()] == ["arc", "ring"]
    assert book.get("ring").sites[0].x_um == 1


def test_store_round_trip(tmp_path):
    state = LayoutState(
        layouts=[
            Layout(
                name="ring",
                sites=[Site(x_um=0.5, y_um=-1.5, on=False)],
                lattice=Lattice(kind="kagome", spacing_um=4.0, sites=12),
                min_spacing_um=2.0,
            )
        ]
    )
    store = LayoutStore(tmp_path / "layouts.json")
    store.save(state)
    assert store.load() == state


def test_close_pairs_of_a_large_array_use_little_memory():
    import tracemalloc

    sites = rectangle(45, 45, 5.0)
    sites[1] = Site(x_um=sites[0].x_um + 1.0, y_um=sites[0].y_um)

    tracemalloc.start()
    pairs = close_pairs(sites, 3.0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert pairs == [(0, 1)]
    assert peak < 5e6, "an all-pairs distance matrix would take ~65 MB here"
