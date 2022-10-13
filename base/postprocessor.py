# post processor
from pystar360.base.dataStruct import bboxes_collector


def limit_defect_n(bboxes, **kwargs):
    """To limit number of defect components, default=5"""
    if not bboxes:
        return []

    default_limit_n = 5
    output_bboxes = []
    bboxes_dict = bboxes_collector(bboxes)

    for k, v in bboxes_dict.items():
        # get limit number
        n = kwargs.get(k, default_limit_n)

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
        defect_bboxes.sort(key=lambda x: x.conf_score)
        # limit number of defect components for each type
        for idx, box in enumerate(defect_bboxes):
            if idx >= limit_n:
                box.is_defect = 0
                box.is_3ddefect = 0

        output_bboxes.extend(defect_bboxes)
        output_bboxes.extend(non_defect_bboxes)
    return output_bboxes
