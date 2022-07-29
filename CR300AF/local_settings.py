
import sys
sys.path.append("..")
from pystar360.global_settings import * 
from pystar360.utilities._logger import get_logger

from pathlib import Path
from typing import Union

FOLDER_NAME: str = "CR300AF"

VIZ: bool = True  # visualize output overwritten
VIZ_LV: int = 1  # visualize level overwritten
NEED_HIST: bool = False # need HISTORY image processing or not overwritten
LOCAL_LOG_PATH: Union[str, Path] = LOG_PATH / FOLDER_NAME # log directionary 
LOCAL_OUTPUT_PATH: Union[str, Path] =  OUTPUT_PATH / FOLDER_NAME # output results directionary 
LOCAL_DATASET_PATH: Union[str, Path] = DATASET_PATH / FOLDER_NAME


LOCAL_LOG_PATH.mkdir(exist_ok=True, parents=True)
LOCAL_OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
LOCAL_DATASET_PATH.mkdir(exist_ok=True, parents=True)

# LOGGER = get_logger(FOLDER_NAME, LOCAL_LOG_PATH)