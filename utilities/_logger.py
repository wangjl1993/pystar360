# -*- coding: utf-8 -*-
import os
import sys 
import logging
from pathlib import Path
from sqlite3 import paramstyle

class DataLogger(object):
    """
    Data logging
    """

    def __init__(self, file_name, *args):
        self.file_name = file_name
        self.update(*args, mode="w")

    def update(self, *args, mode="a"):
        with open(self.file_name, mode) as f:
            DataLogger._write(f, *args)

    @staticmethod
    def _write(f, *args):
        string = ",".join(map(str, args)) + "\n"
        f.write(string)
        f.flush()


def get_logger(
    log_name,
    save_path=None,
    l_level=logging.DEBUG
):

    # Create a custom logger
    d_logger = logging.getLogger(log_name)
    d_logger.handlers.clear()
    d_logger.setLevel(l_level)
        
    # Create console handlers
    c_handler = logging.StreamHandler()
    c_handler.setLevel(l_level)
    format = logging.Formatter(
             "|".join(["%(levelname)s", "%(asctime)-.23s", 
             "[%(filename)s%(lineno)d]:%(message)s"])
            ,datefmt="%Y-%m-%d %H:%M:%S")
    c_handler.setFormatter(format)
    d_logger.addHandler(c_handler)
    d_logger.propagate = False
    # d_logger.handlers = [c_handler]

    if save_path:
        # file logger 
        save_path  = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        file_name = save_path / (log_name + ".log")
        f_handler = logging.FileHandler(file_name)
        f_handler.setLevel(l_level)
        d_logger.addHandler(f_handler)

    return d_logger

d_logger = get_logger("DEVLOPMENT_MODE")
