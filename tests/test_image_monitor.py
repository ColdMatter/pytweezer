import numpy as np
import pyqtgraph as pg
import pytest
from PyQt6 import QtCore

from pytweezer.analysis.cloud_fit import fit_cloud
from pytweezer.analysis.roi_slice import RoiSlice
from pytweezer.GUI import applet
from pytweezer.GUI.viewers import data_sidebar, image_monitor
from pytweezer.GUI.viewers.archive import zmq_ROI
from pytweezer.GUI.viewers.image_monitor import camera_roi, overlay_paths

HEAD = {"roi_x0": 100, "roi_y0": 40, "binning": 2}


def framed_roi(rows, cols, axis_order, shape=(64, 48)):
    """A frame placed as ImageDisplay places it, and an ROI over ``rows`` x ``cols`` of it."""
    plot = pg.PlotItem()
    image = pg.ImageItem(np.zeros(shape), axisOrder=axis_order)
    # ImageDisplay stretches the frame into a (cols, rows) rectangle whatever the axis order.
    image.setRect(QtCore.QRectF(0, 0, shape[1], shape[0]))
    plot.addItem(image)
    if axis_order == "col-major":
        corner, extent = (rows[0], cols[0]), (rows[1] - rows[0], cols[1] - cols[0])
    else:
        corner, extent = (cols[0], rows[0]), (cols[1] - cols[0], rows[1] - rows[0])
    roi = pg.ROI((0, 0))
    plot.addItem(roi)
    roi.setPos(image.mapToParent(QtCore.QPointF(*corner)))
    far = image.mapToParent(
        QtCore.QPointF(corner[0] + extent[0], corner[1] + extent[1])
    )
    roi.setSize(far - roi.pos())
    return plot, roi, image


@pytest.mark.parametrize("axis_order", ["col-major", "row-major"])
def test_the_camera_roi_reads_the_frame_pixels_under_the_roi(qapp, axis_order):
    _plot, roi, image = framed_roi(rows=(10, 30), cols=(5, 21), axis_order=axis_order)
    # Frame pixels are binned by 2 and start at sensor pixel (100, 40).
    assert camera_roi(roi, image, HEAD) == (110, 32, 60, 40)


def test_the_camera_roi_stops_at_the_frame_edge(qapp):
    _plot, roi, image = framed_roi(rows=(50, 90), cols=(-8, 10), axis_order="col-major")
    assert camera_roi(roi, image, {}) == (0, 10, 50, 14)


def test_an_roi_off_the_frame_has_no_camera_roi(qapp):
    _plot, roi, image = framed_roi(rows=(70, 90), cols=(5, 10), axis_order="col-major")
    assert camera_roi(roi, image, HEAD) is None


class FakeProps:
    def __init__(self, name=""):
        self.name = name
        self.values = {}

    def get(self, key, default=None):
        return self.values.setdefault(key, default)

    def set(self, key, value):
        self.values[key] = value

    def changes(self, includeparent=True):
        return []


class FakeStream:
    def __init__(self, name, **kwargs):
        self.queue = []

    def subscribe(self, channel):
        pass

    def unsubscribe(self, channel=None):
        pass

    def has_new_data(self, timeout_ms=1):
        return bool(self.queue)

    def recv(self, **kwargs):
        return self.queue.pop(0)


def fake_hubs(monkeypatch):
    for module, name, fake in [
        (applet, "Properties", FakeProps),
        (zmq_ROI, "Properties", FakeProps),
        (image_monitor, "ImageClient", FakeStream),
        (image_monitor, "DataClient", FakeStream),
        (data_sidebar, "DataClient", FakeStream),
    ]:
        monkeypatch.setattr(module, name, fake)


@pytest.fixture
def monitor(qapp, monkeypatch):
    """An Image Monitor with no hubs or Properties server behind it."""
    fake_hubs(monkeypatch)
    return image_monitor.ImageDisplay("Test Monitor")


def synthetic_cloud(
    shape=(120, 160), x0=70.3, y0=50.7, major=14.0, minor=6.0, angle=0.5
):
    rows, cols = np.indices(shape, dtype=float)
    dx, dy = cols + 0.5 - x0, rows + 0.5 - y0
    along = np.cos(angle) * dx + np.sin(angle) * dy
    across = -np.sin(angle) * dx + np.cos(angle) * dy
    return 100.0 + 500.0 * np.exp(-0.5 * ((along / major) ** 2 + (across / minor) ** 2))


BINNED = {"_imgresolution": [1e-6, 3e-6], "_offset": [10, 20]}


def test_overlay_ellipse_centred_on_fit():
    fit = fit_cloud(synthetic_cloud(), resolution=(1e-6, 3e-6), origin=(10, 20))
    head = fit.as_dict() | BINNED | {"rid": 1, "index": 0}

    paths = overlay_paths(head, 5)

    assert len(paths) == 5
    for path in paths:
        centre = path[:, :-1].mean(axis=1)  # the outline closes on its first point
        assert centre == pytest.approx([fit.x0, fit.y0], rel=1e-9)


def test_failed_fit_clears_overlay():
    fit = fit_cloud(synthetic_cloud())
    assert overlay_paths(fit.as_dict() | {"success": False}, 8) == []
    assert overlay_paths({"success": False, "rid": 4, "index": 0}, 8) == []


def feed_frame(monitor, frame, head):
    monitor.imgstream.queue.append(("cam", dict(head), frame))
    monitor.poll()


@pytest.mark.parametrize(
    "head",
    [BINNED, {"_offset": [10, 20]}, {}],
    ids=["scaled and offset", "offset only", "bare"],
)
def test_frame_pixels_sit_at_their_physical_coordinates(monitor, head):
    feed_frame(monitor, np.zeros((64, 48)), head)
    res_x, res_y = head.get("_imgresolution", (1, 1))
    origin_x, origin_y = head.get("_offset", (0, 0))

    for row, col in [(0, 0), (63, 0), (0, 47), (30, 20)]:
        placed = monitor.image_item.mapToParent(QtCore.QPointF(col + 0.5, row + 0.5))
        assert placed.x() == pytest.approx((origin_x + col + 0.5) * res_x)
        assert placed.y() == pytest.approx((origin_y + row + 0.5) * res_y)


def test_image_monitor_draws_row_zero_at_the_top_with_its_data_panel_on(monitor):
    assert monitor.plot.getViewBox().yInverted()
    assert monitor.image_item.axisOrder == "row-major"
    assert not monitor.sidebar.isHidden()


def test_the_data_panel_is_never_shown_as_a_window_while_the_monitor_builds(
    qapp, monkeypatch
):
    fake_hubs(monkeypatch)
    parents_when_shown = []
    set_visible = data_sidebar.DataSidebar.setVisible

    def recording_set_visible(sidebar, visible):
        parents_when_shown.append(sidebar.parent())
        set_visible(sidebar, visible)

    monkeypatch.setattr(data_sidebar.DataSidebar, "setVisible", recording_set_visible)
    image_monitor.ImageDisplay("Test Monitor")
    assert parents_when_shown
    assert all(parent is not None for parent in parents_when_shown)


def test_overlay_follows_the_cloud_tilt_in_the_plot(monitor):
    frame = synthetic_cloud(angle=0.5)
    feed_frame(monitor, frame, BINNED)
    fit = fit_cloud(frame, resolution=(1e-6, 3e-6), origin=(10, 20))
    monitor.overlayclient.queue.append(("cloud_fit", fit.as_dict() | BINNED))

    monitor.update_overlays()

    rows, cols = np.indices(frame.shape)
    weights = (frame - frame.min()).ravel()
    points = [
        monitor.image_item.mapToParent(QtCore.QPointF(c + 0.5, r + 0.5))
        for r, c in zip(rows.ravel(), cols.ravel())
    ]
    pixels = np.array([[p.x(), p.y()] for p in points])
    cloud_axis = major_axis_angle(pixels, weights)
    outline = np.array(monitor.overlay_curves["cloud_fit"][3].getData())
    assert major_axis_angle(outline.T, np.ones(outline.shape[1])) == pytest.approx(
        cloud_axis, abs=0.02
    )


def major_axis_angle(points, weights):
    """Angle of the principal axis of ``points`` (rows of x, y) from +x towards +y."""
    covariance = np.cov(points.T, aweights=weights)
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, np.argmax(values)]
    return float(np.arctan2(major[1], major[0]) % np.pi)


def test_sidebar_keeps_each_streams_latest_message(monitor):
    sidebar = monitor.sidebar
    sidebar.client.queue += [
        ("stats", {"mean": 1.0}),
        ("fit", {"x0": 5.0}),
        ("stats", {"mean": 2.0}),
    ]
    sidebar.update()
    assert sidebar.blocks["stats"].values["mean"].text() == "2"
    assert sidebar.blocks["fit"].values["x0"].text() == "5"

    sidebar.client.queue.append(("fit", {"x0": 6.0}))
    sidebar.update()
    assert sidebar.blocks["stats"].values["mean"].text() == "2"
    assert sidebar.blocks["fit"].values["x0"].text() == "6"


def test_a_roi_over_pixels_is_cropped_from_the_same_pixels_by_roi_slice(monitor):
    shape = (64, 48)
    feed_frame(monitor, np.zeros(shape), BINNED)
    # Rows 10-29 and columns 5-20, inset so the pixel edges are not on a rounding boundary.
    top_left = monitor.image_item.mapToParent(QtCore.QPointF(5.2, 10.2))
    bottom_right = monitor.image_item.mapToParent(QtCore.QPointF(20.8, 29.8))
    bounds = RoiSlice._pixel_bounds(
        (top_left.x(), top_left.y()),
        (bottom_right.x() - top_left.x(), bottom_right.y() - top_left.y()),
        BINNED["_offset"],
        BINNED["_imgresolution"],
        shape,
    )
    assert bounds == (10, 30, 5, 21)
