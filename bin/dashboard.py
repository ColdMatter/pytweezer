import sys
import signal

from PyQt5.QtWidgets import QApplication

from bin.process_manager import ProcessManager


class Dashboard(ProcessManager):
    categories = ["GUI", "Viewer"]
    
    def __init__(self):
        super().__init__("Dashboard")


def main():
    app = QApplication(sys.argv)
    win = Dashboard()
    win.show()

    def on_exit(_signo, _stack_frame):
        print(f"Closing controller")
        win.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_exit)
    app.exec_()


if __name__ == "__main__":
    main()