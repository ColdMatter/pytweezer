import shutil
import os
import socket
import numpy as np
from PIL import Image
from time import sleep

def MeadowTransferPhasemask():
    source = "C:\\Users\\CaFMOT\\OneDrive - Imperial College London\\caftweezers\\MeadowController\\phasemasks\\phasemask.bmp"
    destination = r"\\PH-NFITCH-2\SLMController\phasemask.bmp"
    try:
        shutil.copy(source, destination)
        print("File copied successfully via SMB.")
    except PermissionError:
        print("Windows blocked the file copy. Check Sharing permissions.")

def MeadowSendCommand(command):
    SERVER_IP = '155.198.206.58'  # Replace with Computer A's IP
    PORT = 65432
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if command == 'OPEN':
                s.connect((SERVER_IP, PORT))
                s.sendall(command.encode('utf-8'))
                print('Command OPEN sent. Opening SLM.')
                sleep(5)
            elif command == "UPDATE":
                s.connect((SERVER_IP, PORT))
                s.sendall(command.encode('utf-8'))
                print('Command UPDATE sent. Updating SLM.')
                sleep(1)
            elif command == "CLOSE":
                s.connect((SERVER_IP, PORT))
                s.sendall(command.encode('utf-8'))
                print("Command CLOSE sent. Closing SLM.")
            elif command == "SHUTDOWN":
                s.connect((SERVER_IP, PORT))
                s.sendall(command.encode('utf-8'))
                print("Command SHUTDOWN sent. Shutting down Meadow server. WARNING: connection will be severed. Server must be restarted locally.")
            elif command == "GETIMAGE":
                s.connect((SERVER_IP, PORT))
                s.sendall(command.encode('utf-8'))
                print("Command GETIMAGE sent. Acquiring Marlin image.")
            else:
                print("Unknown command. Command not sent.")
    except Exception as e:
        print(f"Failed to send command '{command}': {e}")

def GetMarlinImage():
    destination = "C:\\Users\\CaFMOT\\OneDrive - Imperial College London\\caftweezers\\MarlinController\\Images\\marlin_image.tiff"
    source = r"\\PH-NFITCH-2\SLMController\marlin_image.tiff"

    MeadowSendCommand('GETIMAGE')
    try:
        shutil.copy(source, destination)
        print("Marlin image acquired.")
    except PermissionError:
        print("Windows blocked the file copy. Check Sharing permissions.")

    sleep(5)
    img = np.asarray(Image.open(destination))
    return img