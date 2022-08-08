
from enum import Enum


from pystar360.utilities.helper import *


class palette(Enum):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    PURPLE = (125, 38, 205)
    YELLOW = (255, 255, 102)

def plt_bboxes_on_img(bboxes, img, img_h, img_w, startline,
                    axis=1, vis_lv=1, resize_ratio=0.1, 
                    default_color=palette.GREEN):
    if not bboxes:
        return

    img = resize_img(img, resize_ratio)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_h, img_w = get_img_size2(img_h, img_w, resize_ratio)
    for b in bboxes:
        
        if b.is_defect != 0 or b.is_3ddefect != 0:
            color = palette.RED
        else:
            color = default_color
        
        points = frame2rect(b.curr_rect, startline, img_h, img_w, axis=axis)
        cv2.rectangle(img, points[0], points[1], color.value, 1)

        if vis_lv < 1:
            text = b.name
        elif vis_lv < 2:
            text = concat_str(b.name, b.index)
        elif vis_lv < 3:
            text = concat_str(b.name, b.index, b.conf_score)
        else:
            text = concat_str(b.name, b.index, b.conf_score)
            proposal_points = frame2rect(b.proposal_rect, startline, img_h, img_w, axis=axis)
            cv2.rectangle(img, proposal_points[0], proposal_points[1], palette.PURPLE.value, 1)
        
        cv2.putText(img, text, points[0], cv2.FONT_HERSHEY_PLAIN, 1.5, color.value)

    return img 