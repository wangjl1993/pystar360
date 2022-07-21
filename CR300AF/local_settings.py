
import sys
sys.path.append("..")
from uni360detection.base.global_settings import *
from uni360detection.utilities._logger import get_logger

from pathlib import Path
from typing import Union

FOLDER_NAME: str = "CR300AF"
VIZ: bool = True  # visualize output 
VIZ_LV: int = 1  # visualize level 
NEED_HIST: bool = False # need HISTORY image processing or not
LOCAL_LOG_PATH: Union[str, Path] = Path.cwd() /  "log" / FOLDER_NAME # log directionary 
LOCAL_OUTPUT_PATH: Union[str, Path] =  Path.cwd() /  "output" / FOLDER_NAME # output results directionary 

LOCAL_LOG_PATH.mkdir(exist_ok=True, parents=True)
LOCAL_OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

# LOGGER = get_logger(FOLDER_NAME, LOCAL_LOG_PATH)