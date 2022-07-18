import sys
sys.path.append("..")
from uni360detection.base.global_settings import *

from pathlib import Path
from typing import Union

FOLDER_NAME: str = "CRH1A_A"
VIZ: bool = True  # visualize output 
VIZ_LV: int = 1  # visualize level 
NEED_HIST: bool = False # need HISTORY image processing or not
LOCAL_LOG_PATH: Union[str, Path] = Path.cwd() /  "log" / FOLDER_NAME # log directionary 
LOCAL_OUTPUT_PATH: Union[str, Path] =  Path.cwd() /  "output" / FOLDER_NAME # output results directionary 