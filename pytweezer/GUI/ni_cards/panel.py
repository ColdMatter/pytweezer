"""The NI Cards tab: every NI channel on every lab PC, by MOTMaster alias."""

from collections.abc import Callable

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QLabel,
    QScrollArea,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pytweezer.configuration.config import HOSTS, get_config
from pytweezer.GUI.components import Region, set_state
from pytweezer.GUI.ni_cards.cells import ChannelCell
from pytweezer.GUI.ni_cards.client import CardsClient
from pytweezer.GUI.ni_cards.flow_grid import CELL_WIDTH, FlowGrid

NI_CARDS_CLASS = "pytweezer.drivers.ni_cards:NICards"
STATE_POLL_MS = 1000
RECONNECT_MS = 3000
READ_POLL_MS = 200
_ERROR_SHOWN_MS = 8000
#: A Region's left and right content margins plus its border.
_REGION_MARGINS = 26
_KINDS = [
    ("ai", "Analogue in"),
    ("ao", "Analogue out"),
    ("di", "Digital in"),
    ("do", "Digital out"),
]


def ni_card_devices(devices_config: dict) -> list[str]:
    """The ``CONFIG["Devices"]`` entries that are ``NICards`` devices, in config order."""
    return [
        name
        for name, entry in devices_config.items()
        if entry.get("class") == NI_CARDS_CLASS
    ]


class Collapsible(QWidget):
    """A toggle button over a body that starts hidden."""

    def __init__(self, title: str, body: QWidget, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)
        self.button = QToolButton()
        self.button.setObjectName("SectionToggle")
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.button.toggled.connect(self._toggle)
        self.body = body
        body.hide()
        layout.addWidget(self.button)
        layout.addWidget(body)

    def is_collapsed(self) -> bool:
        return self.body.isHidden()

    def _toggle(self, open_: bool) -> None:
        self.button.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if open_ else QtCore.Qt.ArrowType.RightArrow
        )
        self.body.setVisible(open_)


class DeviceView(QWidget):
    """One PC's cards, kept live from its ``NICards`` device while visible.

    While the device is unreachable the state poll carries on, slower, as the
    reconnection attempt.
    """

    def __init__(self, name: str, client, parent=None):
        super().__init__(parent)
        self.name = name
        self.client = client
        self.cells: dict[str, ChannelCell] = {}
        self.banners: dict[str, QLabel] = {}
        self.unaliased_sections: dict[str, Collapsible] = {}
        self.watched: set[str] = set()
        self._connected: bool | None = None
        self._state_pending = False
        self._read_pending = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.message = QLabel()
        self.message.setProperty("role", "regionHint")
        self.message.setWordWrap(True)
        self.message.setText(f"Connecting to {name}…")
        self.error = QLabel()
        self.error.setObjectName("StatusLabel")
        self.error.setWordWrap(True)
        self.error.hide()
        self._error_timer = QtCore.QTimer(
            self, singleShot=True, interval=_ERROR_SHOWN_MS
        )
        self._error_timer.timeout.connect(self.error.hide)
        self.notices = QLabel()
        self.notices.setProperty("role", "regionHint")
        self.notices.setWordWrap(True)
        self.notices.hide()
        layout.addWidget(self.message)
        layout.addWidget(self.notices)
        layout.addWidget(self.error)

        self.content = QWidget()
        self.content.setObjectName("NICardsContent")
        self.cards_layout = QVBoxLayout(self.content)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(10)
        self.cards_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setObjectName("NICardsScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setWidget(self.content)
        scroll.viewport().installEventFilter(self)
        self._viewport = scroll.viewport()
        self.grids: list[FlowGrid] = []
        layout.addWidget(scroll, 1)

        self.state_timer = QtCore.QTimer(self, interval=STATE_POLL_MS)
        self.state_timer.timeout.connect(self.poll_state)
        self.read_timer = QtCore.QTimer(self, interval=READ_POLL_MS)
        self.read_timer.timeout.connect(self.poll_readings)

        client.connection_changed.connect(self.on_connection)
        client.layout_received.connect(self.on_layout)
        client.state_received.connect(self.on_state)
        client.values_received.connect(self.on_values)
        client.failed.connect(self.on_failed)
        client.fetch_layout()

    def on_connection(self, connected: bool) -> None:
        was_connected, self._connected = self._connected, connected
        self._state_pending = self._read_pending = False
        if connected:
            if was_connected is False:
                self.client.fetch_layout()
        else:
            self.message.setText(
                f"{self.name} is not reachable. Start it in the Devices tab."
            )
            self.message.show()
        self.content.setEnabled(connected)
        self._update_timers()

    def on_layout(self, layout: dict) -> None:
        self._clear_cards()
        if layout["error"]:
            self.message.setText(layout["error"])
            self.message.show()
            return
        if not layout["cards"]:
            self.message.setText(f"{self.name}'s hardware file declares no boards.")
            self.message.show()
            return
        self.message.hide()
        for card in layout["cards"]:
            self.cards_layout.insertWidget(
                self.cards_layout.count() - 1, self._card(card)
            )
        self.poll_state()

    def _card(self, card: dict) -> Region:
        hint = card["address"] + (f" · {card['model']}" if card["model"] else "")
        if card["error"]:
            hint += f" · {card['error']}"
        region = Region("well", card["board"], hint)
        banner = QLabel()
        banner.setObjectName("CardBanner")
        banner.hide()
        region.body.addWidget(banner)
        self.banners[card["address"]] = banner

        kinds = [
            kind
            for kind, _ in _KINDS
            if any(c["kind"] == kind for c in card["channels"])
        ]
        for kind, label in _KINDS:
            if kind not in kinds:
                continue
            if len(kinds) > 1:
                heading = QLabel(label)
                heading.setProperty("role", "cellSection")
                region.body.addWidget(heading)
            region.body.addWidget(
                self._grid([c for c in card["channels"] if c["kind"] == kind])
            )
        if card["unaliased"]:
            section = Collapsible(
                f"Unaliased channels ({len(card['unaliased'])})",
                self._grid(card["unaliased"]),
            )
            region.body.addWidget(section)
            self.unaliased_sections[card["board"]] = section
        if card["error"]:
            for channel in card["channels"] + card["unaliased"]:
                self.cells[channel["physical"]].setEnabled(False)
        return region

    def _grid(self, channels: list[dict]) -> FlowGrid:
        grid = FlowGrid()
        grid.set_columns(self._columns())
        self.grids.append(grid)
        for channel in channels:
            cell = ChannelCell(channel)
            cell.write_requested.connect(self.on_write)
            cell.watch_toggled.connect(self.on_watch)
            self.cells[channel["physical"]] = cell
            grid.add(cell)
        return grid

    def _clear_cards(self) -> None:
        while self.cards_layout.count() > 1:
            widget = self.cards_layout.takeAt(0).widget()
            widget.deleteLater()
        self.cells.clear()
        self.grids.clear()
        self.banners.clear()
        self.unaliased_sections.clear()
        self.watched.clear()
        self._update_timers()

    def _columns(self) -> int:
        return (self._viewport.width() - _REGION_MARGINS) // CELL_WIDTH

    def eventFilter(self, watched, event) -> bool:
        if watched is self._viewport and event.type() == QtCore.QEvent.Type.Resize:
            for grid in self.grids:
                grid.set_columns(self._columns())
        return False

    def on_state(self, state: dict) -> None:
        self._state_pending = False
        holds, last_set = state["holds"], state["last_set"]
        for physical, cell in self.cells.items():
            cell.set_hold(holds.get(physical))
            cell.set_last(last_set.get(physical))
        for address, banner in self.banners.items():
            holder = state["card_holds"].get(address)
            banner.setText(f"Held by {holder}" if holder else "")
            banner.setVisible(holder is not None)
        notices = state.get("notices", [])
        self.notices.setText("\n".join(notices))
        self.notices.setVisible(bool(notices))
        self.watched -= set(holds)
        self._update_timers()

    def on_values(self, values: dict) -> None:
        self._read_pending = False
        for physical, value in values.items():
            if physical in self.cells:
                self.cells[physical].set_reading(value)

    def on_failed(self, method: str, message: str) -> None:
        if method == "state":
            self._state_pending = False
        elif method == "read":
            self._read_pending = False
            for physical in self.watched:
                self.cells[physical].set_watching(False)
            self.watched.clear()
            self._update_timers()
            message = f"Stopped watching: {message}"
        self.error.setText(message)
        set_state(self.error, "crashed")
        self.error.show()
        self._error_timer.start()

    def on_write(self, physical: str, value) -> None:
        if self.cells[physical].kind == "do":
            self.client.write_do(physical, bool(value))
        else:
            self.client.write_ao(physical, float(value))

    def on_watch(self, physical: str, watching: bool) -> None:
        if watching:
            self.watched.add(physical)
        else:
            self.watched.discard(physical)
        self._update_timers()

    def poll_state(self) -> None:
        if self._connected is not None and not self._state_pending:
            self._state_pending = True
            self.client.fetch_state()

    def poll_readings(self) -> None:
        self.watched = {p for p in self.watched if self.cells[p].holder is None}
        if not self.watched:
            self._update_timers()
            return
        if not self._read_pending:
            self._read_pending = True
            self.client.read(sorted(self.watched))

    def _update_timers(self) -> None:
        visible = self.isVisible()
        interval = STATE_POLL_MS if self._connected else RECONNECT_MS
        if self.state_timer.interval() != interval:
            self.state_timer.setInterval(interval)
        if visible and self._connected is not None:
            if not self.state_timer.isActive():
                self.state_timer.start()
        else:
            self.state_timer.stop()
        if visible and self._connected and self.watched:
            if not self.read_timer.isActive():
                self.read_timer.start()
        else:
            self.read_timer.stop()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.poll_state()
        self._update_timers()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._update_timers()


def _host_name(device: str) -> str:
    address = get_config()["Devices"].get(device, {}).get("host")
    return next((name for name, ip in HOSTS.items() if ip == address), address or "?")


class NICardsPanel(QTabWidget):
    """One sub-tab per ``NICards`` device in ``CONFIG["Devices"]``."""

    def __init__(
        self,
        devices: list[str] | None = None,
        make_client: Callable[[str], CardsClient] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        if devices is None:
            devices = ni_card_devices(get_config()["Devices"])
        make_client = make_client or CardsClient
        self.views: dict[str, DeviceView] = {}
        for name in devices:
            view = DeviceView(name, make_client(name))
            self.views[name] = view
            self.addTab(view, f"{_host_name(name)} — {name}")
        if not devices:
            empty = QLabel(
                "No NI cards are configured. Add an NICards entry to CONFIG['Devices']."
            )
            empty.setProperty("role", "regionHint")
            self.addTab(empty, "NI Cards")

    def closeEvent(self, event) -> None:
        for view in self.views.values():
            view.client.close()
        super().closeEvent(event)
