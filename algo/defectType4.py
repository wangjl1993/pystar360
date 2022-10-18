################################################################################
#### 松动（3D）类故障算法汇总
################################################################################

import cv2
import copy

from pystar360.algo.algoBase import algoBaseABC, algoDecorator
from pystar360.yolo.inference import YoloInfer, yolo_xywh2xyxy_v2
from pystar360.utilities.helper import crop_segmented_rect, frame2rect
from pystar360.utilities._logger import d_logger


@algoDecorator
class DetectBoltLoose(algoBaseABC):
    """检测--螺栓/螺杆--松动"""

    def __call__(self):
        pass


@algoDecorator
class DetetNutLoose(algoBaseABC):
    """检测--螺帽/螺帽--松动"""

    def __call__(self):
        pass
