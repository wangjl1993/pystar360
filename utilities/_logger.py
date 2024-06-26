# -*- coding: utf-8 -*-
import logging
from pathlib import Path


def get_logger(log_name, save_path=None, l_level=logging.DEBUG, do_simple=False):

    # Create a custom logger
    d_logger = logging.getLogger(log_name)
    d_logger.handlers.clear()
    d_logger.setLevel(l_level)

    # Create console handlers
    c_handler = logging.StreamHandler()
    c_handler.setLevel(l_level)
    if do_simple:
        format = logging.Formatter(
            "|".join(["%(levelname)s", "%(asctime)-.23s", ":%(message)s"]), datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        format = logging.Formatter(
            "|".join(["%(levelname)s", "%(asctime)-.23s", "[%(filename)s%(lineno)d]:%(message)s"]),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    c_handler.setFormatter(format)
    d_logger.addHandler(c_handler)
    d_logger.propagate = False
    # d_logger.handlers = [c_handler]

    if save_path:
        # file logger
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        file_name = save_path / (log_name + ".log")
        f_handler = logging.FileHandler(file_name)
        f_handler.setLevel(l_level)
        d_logger.addHandler(f_handler)

    return d_logger


# 这个是开发模式下的使用的logger，用于debug/development
d_logger = get_logger("DEVLOPMENT_MODE")

def log_info(self_logger, msg):
    if self_logger:
        self_logger.info(msg)
    else:
        d_logger.info(msg)

