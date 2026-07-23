import time

import serial


class ArduinoPulser:
    """Serial front end for the trigger-pulse Arduino.

    Firmware is at ``pytweezer/arduino_pulser/arduino_pulser.ino``; the baud rate
    and the reply format here must match the sketch actually flashed to the
    board.
    """

    def __init__(self, port='COM4', baudrate=250000, timeout=5):

        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self.startup_message = None

    def connect(self, boot_poll_interval=0.1):
        
        self.serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)

        
        while self.serial.in_waiting == 0:
            time.sleep(boot_poll_interval)

        self.startup_message = self.serial.readline().decode('utf-8').strip()
        return self.startup_message

    def send_pulses(self, number_of_pulses, period_us=None):
        if self.serial is None or not self.serial.is_open:
            raise RuntimeError("Not connected. Call connect() first.")

        if period_us is None:
            command = f"{number_of_pulses}\n"
        else:
            command = f"{number_of_pulses},{int(period_us)}\n"
        self.serial.write(command.encode('utf-8'))

        reply = self.serial.readline().decode('utf-8').strip()
        if not reply:
            raise RuntimeError(
                f"No reply within {self.timeout}s for {number_of_pulses} pulses. "
            )
        if not reply.startswith("OK "):
            raise RuntimeError(f"Arduino rejected {command.strip()!r}: {reply}")

        _, _count, span_us = reply.split()
        return int(span_us) / 1e6

    def close(self):
        ''' close the serial port '''
        if self.serial is None or not self.serial.is_open:
            return None

        self.serial.write(b'q\n')
        shutdown_message = self.serial.readline().decode('utf-8').strip()
        self.serial.close()
        return shutdown_message

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


if __name__ == "__main__":
    arduino = ArduinoPulser('COM4')

    startup_message = arduino.connect()
    print(f"Connection Confirmed - Arduino: '{startup_message}'")

    span_s = arduino.send_pulses(10)
    print(f"10 pulses spanned {span_s*1e3:.3f} ms (Arduino clock)")

    shutdown_message = arduino.close()
    print(f"Arduino: '{shutdown_message}'")
    print("Port closed successfully.")
