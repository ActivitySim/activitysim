import os
_directory = os.getcwd()

def get_dir():
    global _directory
    return _directory

def set_dir(directory):
    global _directory
    if directory:
        _directory = directory
    else:
        _directory = os.getcwd()

