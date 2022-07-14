# -*- coding: utf-8 -*-
import os
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
    log_console=True,
):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(process)d:%(levelname)s:%(asctime)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Create a custom logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    if log_console:
        # Create handlers
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)
        # c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        # c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

    if save_path:
        save_path  = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        file_name = save_path / (log_name + ".log")
        f_handler = logging.FileHandler(file_name)
        f_handler.setLevel(logging.DEBUG)
        logger.addHandler(f_handler)
    return logger