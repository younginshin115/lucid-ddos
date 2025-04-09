import time
import os

def write_log(message: str, output_folder: str, prefix: str = ''):
    """
    Prints and appends a message to the history.log file in the given folder.

    Args:
        message (str): The log message to print and write
        output_folder (str): Path to the folder where history.log should be saved
        prefix (str): Optional prefix (like timestamp)
    """
    if prefix:
        message = f"{prefix} | {message}"
    print(message)
    with open(os.path.join(output_folder, 'history.log'), 'a') as f:
        f.write(message + '\n')


def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")
