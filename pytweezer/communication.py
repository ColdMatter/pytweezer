from paramiko import SSHClient, SFTPClient, AutoAddPolicy
import os
import socket
import numpy as np
from PIL import Image
from time import sleep

def MeadowSFTPTransferDirectory(source_dir, target_dir):
    meadowhost = '155.198.206.58'
    username = 'marlin'
    password = 'UltraCold59e'
    meadowSSH = SSHClient()
    meadowSSH.set_missing_host_key_policy(AutoAddPolicy())
    meadowSSH.connect(meadowhost, username=username, password=password)
    meadowSFTP = meadowSSH.open_sftp()
    for item in os.listdir(source_dir):
        if os.path.isfile(os.path.join(source_dir, item)):
            meadowSFTP.put(os.path.join(source_dir, item), '%s/%s' % (target_dir, item))
            print(f'File {target_dir + str(item)} transferred.')
        else:
            meadowSFTP.mkdir(os.path.join(target_dir, item))
            print(f'Directory {target_dir + str(item)} created.')
            SFTP_transfer_directory(sftpclient, os.path.join(source_dir, item), os.path.join(target_dir, item))

def GetMarlinImage(num_images = 10):
    marlinhost = '155.198.206.58'
    username = 'marlin'
    password = 'UltraCold59e'
    local_path ="C:\\Users\\CaFMOT\\OneDrive - Imperial College London\\caftweezers\\MarlinController\\Images\\"
    remote_path = "C:\\Users\\BvSsh_VirtualUsers\\marlin\\"

    # Open SSH connection
    marlinSSH = SSHClient()
    marlinSSH.set_missing_host_key_policy(AutoAddPolicy())
    marlinSSH.connect(marlinhost, username=username, password=password)
    marlinSFTP = marlinSSH.open_sftp()
    print('SSH and SFTP connection established.')

    # Execute SSH command to trigger camera - print process output
    cmd = f"cd marlin; .venv\\Scripts\\activate; python marlin_image.py {num_images}" 
    stdin, stdout, stderror = marlinSSH.exec_command(cmd)
    print(stdout.read().decode())

    # Execute SFTP file transfer of image from remote to local
    filename = 'marlin_image.tiff'
    marlinSFTP.get(remote_path + filename, local_path + filename)
    print('Marlin image transfer complete.')

    # Close SSH connection
    marlinSSH.close()

    img = np.asarray(Image.open(local_path + filename))
    return img

def UpdateMarlinScript():
    marlinhost = '155.198.206.58'
    username = 'marlin'
    password = 'UltraCold59e'
    local_path ="C:\\Users\\CaFMOT\\OneDrive - Imperial College London\\caftweezers\\MarlinController\\Array Monitor\\"
    remote_path = "C:\\Users\\BvSsh_VirtualUsers\\marlin\\"

    # Open SSH connection
    marlinSSH = SSHClient()
    marlinSSH.set_missing_host_key_policy(AutoAddPolicy())
    marlinSSH.connect(marlinhost, username=username, password=password)
    marlinSFTP = marlinSSH.open_sftp()
    print('SSH and SFTP connection established.')

    # Execute SFTP file transfer of image from remote to local
    filename = 'marlin_image.py'
    marlinSFTP.put(local_path + filename, remote_path + filename)
    print('Marlin script transfer complete.')

def MeadowTransferPhasemask():
    meadowhost = '155.198.206.58'
    username = 'marlin'
    password = 'UltraCold59e'
    meadowSSH = SSHClient()
    meadowSSH.set_missing_host_key_policy(AutoAddPolicy())
    meadowSSH.connect(meadowhost, username=username, password=password)
    meadowSFTP = meadowSSH.open_sftp()
    source_dir = "C:\\Users\\CaFMOT\\OneDrive - Imperial College London\\caftweezers\\MeadowController\\phasemasks\\"
    target_dir = "C:\\Program Files\\Meadowlark Optics\\Blink 1920 HDMI\\SLMController\\"
    filename = "phasemask.bmp"
    meadowSFTP.put(source_dir+filename, target_dir+filename)
    meadowSSH.close()

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
            else:
                print("Unknown command. Command not sent.")
    except Exception as e:
        print(f"Failed to send command '{command}': {e}")