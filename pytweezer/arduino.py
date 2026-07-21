import time

import serial


class ArduinoPulser:
    

    def __init__(self, port='COM4', baudrate=9600, timeout=1):
        
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

    def send_pulses(self, number_of_pulses):
        """Send a number of pulses to the Arduino."""
        if self.serial is None or not self.serial.is_open:
            raise RuntimeError("Not connected. Call connect() first.")

        command = f"{number_of_pulses}\n"
        self.serial.write(command.encode('utf-8'))

        confirmation = self.serial.readline().decode('utf-8').strip()
        return confirmation

    def close(self):
        ''' close the serial port '''
        if self.serial is None or not self.serial.is_open:
            return None

        self.serial.write(b'q')
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

    confirmation = arduino.send_pulses(10)
    print(confirmation)

    shutdown_message = arduino.close()
    print(f"Arduino: '{shutdown_message}'")
    print("Port closed successfully.")
