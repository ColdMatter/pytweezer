import sys
import zmq
import json
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QScrollArea, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt


# inside your poll_images function:

ZMQ_PORT = 5555


class StackedViewer(QWidget):
    def __init__(self, subscriber, group_size=2):
        super().__init__()
        self.subscriber = subscriber
        self.group_size = group_size
        self.setWindowTitle("ZeroMQ Stacked Viewer")
        self.resize(800, 900)

        # Scrollable vertical stack
        self.stack_widget = QWidget()
        self.stack_layout = QVBoxLayout(self.stack_widget)
        self.stack_layout.setContentsMargins(5, 5, 5, 5)
        self.stack_layout.setSpacing(5)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.stack_widget)

        layout = QVBoxLayout()
        layout.addWidget(self.scroll)
        self.setLayout(layout)

        # store images until a full group is ready
        self.buffer = []
        self.timer = QTimer()
        self.timer.timeout.connect(self.poll_images)
        self.timer.start(10)

    def poll_images(self):
        """Poll ZeroMQ for new images without blocking."""
        while True:
            try:
                msg_parts = self.subscriber.recv_multipart(flags=zmq.NOBLOCK)
                header_bytes, img_bytes = msg_parts
                # decode metadata
                meta = json.loads(header_bytes.decode("utf-8"))
                shape = tuple(meta["shape"])
                dtype = np.dtype(meta["dtype"])

                # reconstruct array
                arr = np.frombuffer(img_bytes, dtype=dtype).reshape(shape)
                self.buffer.append(arr)
                if len(self.buffer) >= self.group_size:
                    self.show_group(self.buffer[: self.group_size])
                    self.buffer = self.buffer[self.group_size :]
            except zmq.Again:
                break

    def show_group(self, group):
        """Display a group of images stacked vertically, auto-resized to fit."""
        # clear previous stack
        for i in reversed(range(self.stack_layout.count())):
            widget = self.stack_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        if not group:
            return

        # compute available height per image
        total_height = self.scroll.viewport().height()
        per_image_height = total_height // len(group)
        viewport_width = self.scroll.viewport().width()

        for arr in group:
            h, w, c = arr.shape
            qimg = QImage(arr.data, w, h, c * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            # scale to fit width and per-image height
            scaled_pixmap = pixmap.scaled(
                viewport_width,
                per_image_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

            label = QLabel()
            label.setPixmap(scaled_pixmap)
            label.setAlignment(Qt.AlignCenter)
            self.stack_layout.addWidget(label)

        # scroll to bottom (optional, for latest group)
        self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()
        )


def run_viewer(group_size):
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(f"tcp://127.0.0.1:{ZMQ_PORT}")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    app = QApplication(sys.argv)
    viewer = StackedViewer(subscriber, group_size=group_size)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_viewer()
