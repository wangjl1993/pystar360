# post processor

import numpy as np
from pystar360.base.dataStruct import bboxes_collector


def limit_defect_n(bboxes, kwargs):
    """To limit number of defect components
    kwargs = {"label_name": number; "label_name2": number2}
    """
    if not bboxes:
        return []

    default_limit_mu = 4
    default_limit_std = 1
    output_bboxes = []
    bboxes_dict = bboxes_collector(bboxes)

    for k, v in bboxes_dict.items():
        # 如果在限制清单里面
        if k in kwargs:
            # get limit number randomly
            n = kwargs.get(k, default_limit_mu)
            n = int(max(np.random.normal(n, default_limit_std), 0))

            # separate defect bboxes and non-defect bboxes
            defect_bboxes = []
            non_defect_bboxes = []
            for box in v:
                if box.is_defect or box.is_3ddefect:
                    defect_bboxes.append(box)
                else:
                    non_defect_bboxes.append(box)

            # sort according to score
            limit_n = min(len(defect_bboxes), n)
            defect_bboxes.sort(key=lambda x: (x.conf_score, x.value_3d))
            # limit number of defect components for each type
            for idx, box in enumerate(defect_bboxes):
                if idx >= limit_n:
                    box.is_defect = 0
                    box.is_3ddefect = 0

            output_bboxes.extend(defect_bboxes)
            output_bboxes.extend(non_defect_bboxes)
        else:
            # if it is not required to limit by n, then extend
            output_bboxes.extend(v)

    return output_bboxes
