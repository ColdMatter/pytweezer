"""Live loading histograms and loading map of the tweezer array.

Shows what ``analysis.get_array_loading_statistics(show_histogram=True)``
plots in the optimisation notebooks, updating on every shot: the photon
histogram of each site, the histogram of all sites with its two-Gaussian fit
and occupancy threshold, and each site's loading probability at its place in
the image. A line beneath gives the mean loading, its spread over sites, the
fidelity and the fit.

Subscribe it to a ``tweezer_loading.py`` analysis stream. The loading map's
colour range can be dragged on its colour bar.

Properties:
    *   datastreams: ([str]) input data streams.
    *   bins: (int) bins of the all-sites histogram.
    *   site_bins: (int) bins of each site's histogram.
    *   show_site_labels: (bool) number the sites on the loading map.
"""

from PyQt6 import QtWidgets

from pytweezer.GUI.applet import Applet, run_applet
from pytweezer.GUI.tweezer_loading_plots import TweezerLoadingPlots
from pytweezer.servers import DataClient, PropertyAttribute


class TweezerLoadingViewer(Applet):
    """Per-site and all-site photon histograms beside the array's loading map."""

    stream_category = "Data"
    poll_interval = 100
    default_size = (1500, 500)

    _datastreams = PropertyAttribute("datastreams", ["--None--"])
    _bins = PropertyAttribute("bins", 60)
    _site_bins = PropertyAttribute("site_bins", 40)
    _show_site_labels = PropertyAttribute("show_site_labels", False)

    def init_gui(self):
        self.stream = DataClient(self.name)
        self.latest = None

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.plots = TweezerLoadingPlots()
        layout.addWidget(self.plots)

        for plot in self.plots.plot_items:
            menu = plot.getViewBox().menu
            menu.addAction("subscriptions").triggered.connect(
                lambda: self.open_subscription_editor()
            )
            menu.addAction("configure").triggered.connect(self.open_config_editor)

        self.update_subscriptions()

    def update_subscriptions(self):
        """Re-read the properties and redraw the last message against them."""
        self.stream.unsubscribe()
        self.streams = [name for name in self._datastreams if name != "--None--"]
        self.stream.subscribe(self.streams)
        self.plots.bins = int(self._bins)
        self.plots.site_bins = int(self._site_bins)
        self.plots.show_site_labels = bool(self._show_site_labels)
        if not self.streams:
            self.plots.show_message(
                "Subscribe to a tweezer_loading analysis stream (right-click a plot)"
            )
        elif self.latest is None:
            self.plots.show_message(f"Waiting for shots on {', '.join(self.streams)}")
        else:
            self.plots.show_result(*self.latest)

    def poll(self):
        latest = None
        # Only the newest message is drawn: each one already holds the whole window.
        while self.stream.has_new_data():
            message = self.stream.recv()
            if message is None or len(message) < 2 or message[0] not in self.streams:
                continue
            latest = (message[1], message[2] if len(message) > 2 else None)
        if latest is not None:
            self.latest = latest
            self.plots.show_result(*latest)


def main(name):
    run_applet(TweezerLoadingViewer, default_name=name)


if __name__ == "__main__":
    run_applet(TweezerLoadingViewer, default_name="Tweezer Loading")
