
from pathlib import Path
from typing import Union

VIZ: bool = True  # visualize output 
VIZ_LV: int = 1  # visualize level 
NEED_HIST: bool = False # need HISTORY image processing or not
TRAIN_LIB_PATH: Union[str, Path] = Path.cwd() / "uni360detection" / "train_library"
OUTPUT_PATH:  Union[str, Path] = Path.cwd() / "output"
LOG_PATH: Union[str, Path] = Path.cwd() / "log"
DATASET_PATH: Union[str, Path] = Path.cwd() / "dataset"
