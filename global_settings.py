from pathlib import Path
from typing import Union, Optional

# basic settings
MODULE_NAME = Path(__file__).parent.name
DEV: bool = False  # if it's in development mode
VIZ: bool = False  # visualize output
VIZ_LV: int = 1  # visualize level
NEED_HIST: bool = False  # need HISTORY image processing or not
TRAIN_LIB_PATH: Union[str, Path] = Path.cwd() / MODULE_NAME / "train_library"
OUTPUT_PATH: Union[str, Path] = Path.cwd() / "OUTPUT"
LOG_PATH: Union[str, Path] = Path.cwd() / "LOG"
DATASET_PATH: Union[str, Path] = Path.cwd() / "DATASET"
WEIGHT_PATH: Union[str, Path] = Path.cwd() / "weights"

# machine code after encrption
from pystar360.utilities.de import get_mac_password

# set do_decrpt=False, force program to read raw files
MAC_PASSWORD: Optional[str] = get_mac_password(do_decrpt=False)

# print product information
from pystar360.utilities.misc import print_product_info

print_product_info()

# constant
# 不同客户可能有不同的版本
# HUAXING 华兴
# ZHUDAO 主导
# HITO 海拓
CLIENT = "HITO"
SUFFIX_3D = "3D"
USE_3D = False
USE_HIST = False