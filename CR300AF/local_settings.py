
import sys
sys.path.append("..")
from pystar360.global_settings import * 
from pystar360.utilities._logger import get_logger

from pathlib import Path
from typing import Union

FOLDER_NAME: str = Path(__file__).parent.name #"CR300AF"

VIZ: bool = True  # visualize output overwritten
VIZ_LV: int = 1  # visualize level overwritten
NEED_HIST: bool = False # need HISTORY image processing or not overwritten
LOG_PATH: Union[str, Path] = LOG_PATH / FOLDER_NAME # log directionary 
OUTPUT_PATH: Union[str, Path] =  OUTPUT_PATH / FOLDER_NAME # output results directionary 
DATASET_PATH: Union[str, Path] = DATASET_PATH / FOLDER_NAME


LOG_PATH.mkdir(exist_ok=True, parents=True)
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
DATASET_PATH.mkdir(exist_ok=True, parents=True)

# LOGGER = get_logger(FOLDER_NAME, LOCAL_LOG_PATH)