"""Draw a tweezer array on the focal plane and save it as a named layout.

Sites are placed by tapping: in Grid mode a tap snaps to the lattice of the
last preset, in Freehand mode it lands where tapped. Tapping a site switches it
on or off. Layouts are saved to the Experiment Manager (see
:mod:`pytweezer.experiment.layouts`), and "Put on SLM" queues
``SlmTweezerLayout`` with the open layout, with the other arguments of the last
``SlmTweezerLayout`` (or else ``SlmTweezerArray``) run. Its saved-backed optics
are left out, so they are read fresh from the saved parameters.

Coordinates are µm in the focal plane, relative to the blaze spot, but the
canvas is drawn turned 180°, as the Tweezer Monitor shows the HamCam image,
so a pattern looks the way the atoms will. The dashed square is the field the
SLM can steer light into.

Single taps do everything, so it works on a touchscreen; drag pans, the wheel
or the zoom buttons zoom.
"""

from typing import Literal

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from pytweezer.experiment.client import (
    ExperimentManagerClient,
    ManagerError,
    ManagerUnavailable,
    delete_layout,
    get_saved_parameters,
    layouts,
    save_layout,
    submit,
)
from pytweezer.experiment.layouts import (
    Lattice,
    Layout,
    Site,
    close_pairs,
    lattice_points,
    nearest_lattice_point,
    preset_sites,
)
from pytweezer.experiment.task import TaskRequest
from pytweezer.GUI.applet import Applet, run_applet
from pytweezer.GUI.components import CompactDoubleSpinBox, set_state
from pytweezer.GUI.theme import (
    CURVE_COLORS,
    PLOT_FOREGROUND,
    PLOT_SEPARATOR,
    STATE_STYLE,
)
from pytweezer.helper_scripts.array_hologram import reachable_field_um
from pytweezer.logging_utils import get_logger

logger = get_logger("tweezer_layout")

EXPERIMENT_MODULE = "pytweezer.experiments.slm_tweezer_array"
#: Screen pixels within which a tap picks a site.
PICK_RADIUS_PX = 12
#: Above this many lattice points in view the grid dots are not drawn.
MAX_LATTICE_DOTS = 4000
MANAGER_DOWN = "Experiment Manager unreachable: layouts can't be listed or saved"

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


def last_request(class_name: str) -> TaskRequest | None:
    """The last request the Experiment Manager ran for ``class_name`` of the SLM experiments."""
    return ExperimentManagerClient().last_request(EXPERIMENT_MODULE, class_name)


class _LayoutCombo(QtWidgets.QComboBox):
    about_to_open = QtCore.pyqtSignal()

    def showPopup(self):
        self.about_to_open.emit()
        super().showPopup()


class TweezerLayoutApplet(Applet):
    """Draw, save and open tweezer layouts, and put one on the SLM."""

    poll_interval = 0
    default_size = (1000, 820)

    def init_gui(self):
        self.editor = LayoutEditor()
        self.saved_layouts: dict[str, Layout] = {}
        self.field_item = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.addLayout(self._layout_row())
        outer.addLayout(self._preset_row())
        outer.addLayout(self._mode_row())
        self.plot = pg.PlotWidget()
        outer.addWidget(self.plot, stretch=1)
        outer.addLayout(self._action_row())
        self._build_canvas()

        self.refresh_layouts()
        self._draw_field()
        self.redraw()
        self.fit_view()
        if not self.status.text():
            self._say(
                "Open a layout, apply a preset, or pick Freehand and tap to add sites"
            )

    # -- widgets -------------------------------------------------------------

    def _layout_row(self):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Layout"))
        self.layout_combo = _LayoutCombo()
        self.layout_combo.setMinimumWidth(240)
        self.layout_combo.setPlaceholderText("unsaved")
        self.layout_combo.about_to_open.connect(self.refresh_layouts)
        self.layout_combo.activated.connect(self._open_selected)
        row.addWidget(self.layout_combo)
        for text, slot in (("Save", self.save), ("Save as…", self.save_as)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        row.addStretch(1)
        self.delete_button = QtWidgets.QPushButton("Delete")
        self.delete_button.setObjectName("DangerButton")
        self.delete_button.setToolTip("Delete the open layout for every PC")
        self.delete_button.clicked.connect(self.delete)
        row.addWidget(self.delete_button)
        return row

    def _preset_row(self):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Start from"))
        self.preset_kind = QtWidgets.QComboBox()
        self.preset_kind.addItems(["Rectangle", "Kagome"])
        row.addWidget(self.preset_kind)

        rectangle = QtWidgets.QWidget()
        rectangle_row = QtWidgets.QHBoxLayout(rectangle)
        rectangle_row.setContentsMargins(0, 0, 0, 0)
        self.rows_box = self._count_box(15)
        self.columns_box = self._count_box(15)
        rectangle_row.addWidget(self.rows_box)
        rectangle_row.addWidget(QtWidgets.QLabel("×"))
        rectangle_row.addWidget(self.columns_box)
        kagome = QtWidgets.QWidget()
        kagome_row = QtWidgets.QHBoxLayout(kagome)
        kagome_row.setContentsMargins(0, 0, 0, 0)
        self.sites_box = self._count_box(30, maximum=2000)
        kagome_row.addWidget(self.sites_box)
        kagome_row.addWidget(QtWidgets.QLabel("sites"))
        self.preset_counts = QtWidgets.QStackedWidget()
        self.preset_counts.addWidget(rectangle)
        self.preset_counts.addWidget(kagome)
        self.preset_kind.currentIndexChanged.connect(self.preset_counts.setCurrentIndex)
        row.addWidget(self.preset_counts)

        self.spacing_box = self._um_box(5.0, minimum=0.1)
        self.angle_box = CompactDoubleSpinBox()
        self.angle_box.setRange(-180, 180)
        self.angle_box.setDecimals(2)
        self.angle_box.setSuffix(" °")
        self.angle_box.setValue(2.5)
        for label, box in (("spacing", self.spacing_box), ("angle", self.angle_box)):
            row.addWidget(QtWidgets.QLabel(label))
            row.addWidget(box)
        apply = QtWidgets.QPushButton("Apply")
        apply.setToolTip("Replace the sites with this preset, and snap to its lattice")
        apply.clicked.connect(self.apply_preset_from_form)
        row.addWidget(apply)
        row.addStretch(1)
        return row

    def _mode_row(self):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Mode"))
        self.grid_button = QtWidgets.QRadioButton("Grid")
        self.grid_button.setToolTip("A tap adds a site on the nearest lattice point")
        self.freehand_button = QtWidgets.QRadioButton("Freehand")
        self.freehand_button.setToolTip("A tap adds a site exactly where you tap")
        self.grid_button.setChecked(True)
        self.grid_button.toggled.connect(self._mode_changed)
        row.addWidget(self.grid_button)
        row.addWidget(self.freehand_button)
        row.addSpacing(16)
        row.addWidget(QtWidgets.QLabel("Min spacing"))
        self.min_spacing_box = self._um_box(self.editor.min_spacing_um, minimum=0.1)
        self.min_spacing_box.setToolTip(
            "A freehand tap this close to a site toggles it instead of adding one; "
            "closer pairs are outlined red"
        )
        self.min_spacing_box.valueChanged.connect(self._min_spacing_changed)
        row.addWidget(self.min_spacing_box)
        row.addSpacing(16)
        self.counts_label = QtWidgets.QLabel()
        row.addWidget(self.counts_label)
        row.addStretch(1)
        for text, tip, slot in (
            ("−", "Zoom out", lambda: self._zoom(1.25)),
            ("+", "Zoom in", lambda: self._zoom(0.8)),
            ("Fit", "Fit the view to the sites", self.fit_view),
        ):
            button = QtWidgets.QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            row.addWidget(button)
        return row

    def _action_row(self):
        row = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel()
        self.status.setObjectName("StatusLabel")
        row.addWidget(self.status, stretch=1)
        remove = QtWidgets.QPushButton("Remove off sites")
        remove.setToolTip("Delete every switched-off site from the layout")
        remove.clicked.connect(self.remove_off_sites)
        row.addWidget(remove)
        self.put_button = QtWidgets.QPushButton("Put on SLM")
        self.put_button.setObjectName("PrimaryButton")
        self.put_button.setToolTip(
            "Queue SlmTweezerLayout with this layout and the last run's settings"
        )
        self.put_button.clicked.connect(self.put_on_slm)
        row.addWidget(self.put_button)
        return row

    @staticmethod
    def _count_box(value, maximum=200):
        box = QtWidgets.QSpinBox()
        box.setRange(1, maximum)
        box.setValue(value)
        return box

    @staticmethod
    def _um_box(value, minimum):
        box = CompactDoubleSpinBox()
        box.setRange(minimum, 1000.0)
        box.setDecimals(3)
        box.setSuffix(" µm")
        box.setValue(value)
        return box

    def _build_canvas(self):
        item = self.plot.getPlotItem()
        view = item.getViewBox()
        view.setAspectLocked(True)
        view.invertX(True)
        view.invertY(True)
        for side, name in (("bottom", "x"), ("left", "y")):
            item.setLabel(side, name, units="µm")
            item.getAxis(side).enableAutoSIPrefix(False)
        self.lattice_item = pg.ScatterPlotItem(
            size=4, pen=None, brush=pg.mkBrush(PLOT_SEPARATOR)
        )
        self.sites_item = pg.ScatterPlotItem(size=14, pxMode=True)
        item.addItem(self.lattice_item)
        item.addItem(self.sites_item)
        view.sigRangeChanged.connect(lambda *_: self._draw_lattice())
        self.plot.scene().sigMouseClicked.connect(self._clicked)

    # -- drawing -------------------------------------------------------------

    def _draw_field(self):
        try:
            x0, y0, width, height = reachable_field_um(get_saved_parameters())
        except (ManagerUnavailable, ManagerError, KeyError) as error:
            logger.warning("No SLM field outline: %s", error)
            return
        self.field_item = QtWidgets.QGraphicsRectItem(x0, y0, width, height)
        self.field_item.setPen(
            pg.mkPen(PLOT_FOREGROUND, style=QtCore.Qt.PenStyle.DashLine, cosmetic=True)
        )
        self.plot.getPlotItem().addItem(self.field_item)

    def redraw(self):
        close = self.editor.close_site_indices()
        on_brush = pg.mkBrush(CURVE_COLORS[0])
        on_pen = pg.mkPen(CURVE_COLORS[0])
        off_pen = pg.mkPen(PLOT_FOREGROUND, width=1.5)
        close_pen = pg.mkPen(STATE_STYLE["crashed"][0], width=2.5)
        self.sites_item.setData(
            [
                {
                    "pos": (site.x_um, site.y_um),
                    "brush": on_brush if site.on else pg.mkBrush(None),
                    "pen": close_pen if i in close else on_pen if site.on else off_pen,
                }
                for i, site in enumerate(self.editor.sites)
            ]
        )
        on, off = self.editor.counts()
        self.counts_label.setText(f"{on} on · {off} off")
        self.grid_button.setEnabled(self.editor.lattice is not None)
        self.delete_button.setEnabled(self.editor.name in self.saved_layouts)
        title = self.editor.name or "unsaved"
        self.setWindowTitle(f"{self.name}: {title}{' *' if self.editor.dirty else ''}")
        self._draw_lattice()

    def _draw_lattice(self):
        lattice = self.editor.lattice
        if lattice is None or self.editor.mode != "grid":
            self.lattice_item.setData([])
            return
        (x0, x1), (y0, y1) = self.plot.getViewBox().viewRange()
        area_per_point = lattice.spacing_um**2 / (3 if lattice.kind == "kagome" else 1)
        if (x1 - x0) * (y1 - y0) / area_per_point > MAX_LATTICE_DOTS:
            self.lattice_item.setData([])
            return
        points = lattice_points(lattice, (x0, x1), (y0, y1))
        self.lattice_item.setData(pos=points)

    def fit_view(self):
        view = self.plot.getViewBox()
        if self.editor.sites:
            view.autoRange(items=[self.sites_item], padding=0.15)
        elif self.field_item is not None:
            view.autoRange(items=[self.field_item])

    def _zoom(self, factor):
        self.plot.getViewBox().scaleBy((factor, factor))

    def _say(self, text, error=False):
        self.status.setText(text)
        set_state(self.status, "error" if error else "")

    # -- editing -------------------------------------------------------------

    def _clicked(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton or event.double():
            return
        view = self.plot.getViewBox()
        if not view.sceneBoundingRect().contains(event.scenePos()):
            return
        point = view.mapSceneToView(event.scenePos())
        pixel_w, pixel_h = view.viewPixelSize()
        self.editor.tap(
            point.x(), point.y(), pick_radius_um=PICK_RADIUS_PX * max(pixel_w, pixel_h)
        )
        self.redraw()

    def _mode_changed(self):
        self.editor.mode = "grid" if self.grid_button.isChecked() else "freehand"
        self._draw_lattice()

    def _min_spacing_changed(self, value):
        if value != self.editor.min_spacing_um:
            self.editor.min_spacing_um = value
            self.editor.dirty = True
            self.redraw()

    def _discard_edits_ok(self) -> bool:
        return not self.editor.dirty or self.confirm(
            "The open layout has unsaved changes. Discard them?"
        )

    def preset_lattice(self) -> Lattice:
        common = {
            "spacing_um": self.spacing_box.value(),
            "angle_deg": self.angle_box.value(),
        }
        if self.preset_kind.currentText() == "Kagome":
            return Lattice(kind="kagome", sites=self.sites_box.value(), **common)
        return Lattice(
            kind="rectangle",
            rows=self.rows_box.value(),
            columns=self.columns_box.value(),
            **common,
        )

    def apply_preset_from_form(self):
        if not self._discard_edits_ok():
            return
        self.editor.apply_preset(self.preset_lattice())
        self.grid_button.setChecked(True)
        self.redraw()
        self.fit_view()

    def remove_off_sites(self):
        self.editor.remove_off_sites()
        self.redraw()

    def _show_layout(self, layout: Layout):
        self.editor.load(layout)
        self.min_spacing_box.blockSignals(True)
        self.min_spacing_box.setValue(layout.min_spacing_um)
        self.min_spacing_box.blockSignals(False)
        if layout.lattice is None:
            self.freehand_button.setChecked(True)
        self._select(layout.name)
        self.redraw()
        self.fit_view()

    # -- layouts on the manager ---------------------------------------------

    def refresh_layouts(self):
        try:
            found = layouts()
        except (ManagerUnavailable, ManagerError) as error:
            logger.warning("Listing layouts: %s", error)
            self._say(MANAGER_DOWN, error=True)
            return
        self.saved_layouts = {layout.name: layout for layout in found}
        self.layout_combo.blockSignals(True)
        self.layout_combo.clear()
        self.layout_combo.addItems(list(self.saved_layouts))
        self.layout_combo.blockSignals(False)
        self._select(self.editor.name)

    def _select(self, name):
        self.layout_combo.setCurrentIndex(self.layout_combo.findText(name or ""))

    def _open_selected(self, index):
        name = self.layout_combo.itemText(index)
        if name == self.editor.name and not self.editor.dirty:
            return
        if not self._discard_edits_ok():
            self._select(self.editor.name)
            return
        self._show_layout(self.saved_layouts[name])

    def save(self) -> bool:
        if self.editor.name is None:
            return self.save_as()
        return self._save(self.editor.name, overwrite=True)

    def save_as(self) -> bool:
        name = self.ask_name(self.editor.name or "")
        if not name or not name.strip():
            return False
        name = name.strip()
        overwrite = name in self.saved_layouts
        if overwrite and not self.confirm(
            f"A layout {name!r} already exists. Replace it for every PC?"
        ):
            return False
        return self._save(name, overwrite=overwrite)

    def _save(self, name: str, overwrite: bool) -> bool:
        if not any(site.on for site in self.editor.sites):
            self._say("Nothing to save: switch at least one site on", error=True)
            return False
        pairs = len(close_pairs(self.editor.sites, self.editor.min_spacing_um))
        if pairs and not self.confirm(
            f"{pairs} pair{'s' if pairs > 1 else ''} of sites "
            f"{'are' if pairs > 1 else 'is'} closer than the minimum spacing. "
            "Save anyway?"
        ):
            return False
        try:
            save_layout(self.editor.to_layout(name), overwrite=overwrite)
        except (ManagerUnavailable, ManagerError) as error:
            self._say(f"Saving {name!r}: {error}", error=True)
            return False
        self.editor.name = name
        self.editor.dirty = False
        self.refresh_layouts()
        self._say(f"Saved {name!r}")
        self.redraw()
        return True

    def delete(self):
        name = self.editor.name
        if name not in self.saved_layouts or not self.confirm(
            f"Delete layout {name!r} for every PC?"
        ):
            return
        try:
            delete_layout(name)
        except (ManagerUnavailable, ManagerError) as error:
            self._say(f"Deleting {name!r}: {error}", error=True)
            return
        self.editor.name = None
        self.editor.dirty = True
        self.refresh_layouts()
        self._say(f"Deleted {name!r}; its sites are still here, unsaved")
        self.redraw()

    def _slm_arguments(self) -> dict:
        """Arguments of the last SLM run, without the geometry or saved-backed optics."""
        from pytweezer.experiments.slm_tweezer_array import SlmTweezerLayout

        schema = SlmTweezerLayout.schema()["arguments"]
        for class_name in ("SlmTweezerLayout", "SlmTweezerArray"):
            try:
                request = last_request(class_name)
            except (ManagerUnavailable, ManagerError) as error:
                logger.warning("Last %s request: %s", class_name, error)
                return {}
            if request is not None:
                return {
                    name: value
                    for name, value in request.args.items()
                    if name in schema
                    and name != "layout"
                    and not schema[name].get("saved")
                }
        return {}

    def put_on_slm(self):
        if (self.editor.dirty or self.editor.name is None) and not (
            self.confirm("Save the layout before putting it on the SLM?")
            and self.save()
        ):
            return
        name = self.editor.name
        try:
            rid = submit(
                f"{EXPERIMENT_MODULE}:SlmTweezerLayout",
                **self._slm_arguments(),
                layout=name,
            )
        except (ManagerUnavailable, ManagerError, ValueError) as error:
            self._say(f"Putting {name!r} on the SLM: {error}", error=True)
            return
        self._say(f"Queued run {rid}: {name!r} on the SLM")

    # -- dialogs -------------------------------------------------------------

    def confirm(self, question: str) -> bool:
        answer = QtWidgets.QMessageBox.question(self, self.name, question)
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def ask_name(self, current: str) -> str | None:
        name, ok = QtWidgets.QInputDialog.getText(
            self, self.name, "Layout name:", text=current
        )
        return name if ok else None


def main(name):
    run_applet(TweezerLayoutApplet, default_name=name)


if __name__ == "__main__":
    run_applet(TweezerLayoutApplet, default_name="Tweezer Layout")
