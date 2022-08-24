#!/usr/bin/env python
# -*- coding: utf-8 -*-

# start code
import cv2
import numpy as np


class imread_decorator:
    def __init__(self, resize_ratio):
        self.resize_ratio = resize_ratio

    def __call__(self, path, size=None, mode=cv2.IMREAD_GRAYSCALE):
        img = cv2.imread(path, mode)
        if size is None:
            img = cv2.resize(img,
                             None,
                             fx=self.resize_ratio,
                             fy=self.resize_ratio)
        else:
            img = cv2.resize(img, size)
        return img


imread_tenth = imread_decorator(0.1)
imread_octa = imread_decorator(0.125)
imread_quarter = imread_decorator(0.25)
imread_half = imread_decorator(0.5)
imread_full = imread_decorator(1)


def resize_img(img, resize_ratio):
    """Resize image"""
    return cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio)


def concat_str(*args, separator="_"):
    """Concatenate string"""
    args = list(map(str, args))
    return separator.join(args)


def split_str(mystr, separator="="):
    """Split string"""
    return mystr.strip(separator).split(separator)


def get_label_num2check(mystr, separator="="):
    output = split_str(mystr, separator)
    if len(output) == 1:
        return output[0], 1
    elif len(output) == 2:
        return output[0], int(output[1])
    else:
        raise ValueError(f"Please provide a valid label. {mystr} vs {output}")

def frame2index(frame, length):
    """Frame to index and pixel shift"""
    shift = int(frame)
    pixel = int((frame % 1) * length)
    return shift, pixel


def index2frame(shift, pixel, length):
    """Image index and pixel shift to Frame name"""
    return shift + pixel / length


def cutpoints2frame(l, cutpoints, length):
    """filename, pixel to frame point"""
    shift = l.index(cutpoints[0])
    return index2frame(shift, cutpoints[-1], length)


def frame2cutpoints(l, frame, length):
    """frame point to filename, pixel"""
    shift, pixel = frame2index(frame, length)
    return (l[shift], pixel)


def get_img_size(f, img_read):
    """return image size"""
    return img_read(f).shape


def get_img_size2(h, w, resize_ratio):
    """return image size according to a given ratio"""
    h = int(h * resize_ratio)
    w = int(w * resize_ratio)
    return h, w


def read_segmented_img(l, startline, endline, imread, imgsz=None, axis=1):
    """Read continuous image frame and concatenate into one image"""
    assert startline < endline, "startline <= endline"
    if imgsz:
        img_h, img_w = imgsz
        img_h, img_w = get_img_size2(img_h, img_w, imread.resize_ratio)
    else:
        img_h, img_w = imread(l[0]).shape

    if axis == 0:
        start_idx, start_p = frame2index(startline, img_h)
        end_idx, end_p = frame2index(endline, img_h)
    else:
        start_idx, start_p = frame2index(startline, img_w)
        end_idx, end_p = frame2index(endline, img_w)

    start_f = l[start_idx]
    end_f = l[end_idx]

    if axis == 0:
        img = imread(start_f)[start_p:, :]
    else:
        img = imread(start_f)[:, start_p:]

    if start_idx != end_idx:
        tmp_l = l[start_idx + 1:end_idx]
        for f in tmp_l:
            tmp_img = imread(f)
            if axis == 0:
                if tmp_img.shape[0] < img_h:
                    addition = np.zeros((img_h - tmp_img.shape[0], img_w))
                    tmp_img = np.concatenate((tmp_img, addition), axis=axis)
            else:
                if tmp_img.shape[1] < img_w:
                    addition = np.zeros((img_h, img_w - tmp_img.shape[1]))
                    tmp_img = np.concatenate((tmp_img, addition), axis=axis)

            img = np.concatenate((img, tmp_img), axis=axis)

        tmp_img = imread(end_f)
        img = np.concatenate((img, tmp_img[:end_p, :]), axis=axis) if axis == 0 \
        else np.concatenate((img, tmp_img[:, :end_p]), axis=axis)

    else:
        tmp_img = imread(end_f)
        img = tmp_img[
            start_p:end_p, :] if axis == 0 else tmp_img[:, start_p:end_p]

    return img


def xyxy2xywh(points):
    """left-top right-bottom rect points to center xy width height rect points"""
    h = points[1][1] - points[0][1]
    w = points[1][0] - points[0][0]
    x = 0.5 * (points[1][0] + points[0][0])
    y = 0.5 * (points[1][1] + points[0][1])
    return x, y, w, h


def xyxy2left_xywh(points):
    """left-top right-bottom rect points to left-top width height rect points"""
    h = int(points[1][1] - points[0][1])
    w = int(points[1][0] - points[0][0])
    return points[0][0], points[0][1], w, h


def x2pixel(x, sx, img_w, axis=1):
    """frame values (float) to pixel (int) in x direction"""
    if axis == 0:
        o = int(np.ceil((x - sx) * img_w))
        o = min(max(o, 0), img_w - 1) 
    else:
        o = int(np.ceil((x - sx) * img_w))
        o = max(o, 0)
    return o


def y2pixel(y, sy, img_h, axis=1):
    """frame values (float) to pixel (int) in y direction"""
    if axis == 0:
        o = int(np.ceil((y - sy) * img_h))
        o = max(o, 0) 
    else:
        o = int(np.ceil((y - sy) * img_h))
        o = min(max(o, 0), img_h - 1)
    return o


def frame2rect(points, start_main_axis_fp, img_h, img_w,  start_minor_axis_fp=0, axis=1):
    """frame values (float) to pixel (int) in both horizontal and vertical direction
    start_main_axis_fp: x startline if x-axis is the main direction / y startline if y-axis is the main direction
    """
    if axis == 0:
        sx = start_minor_axis_fp
        sy = start_main_axis_fp
    else:
        sx = start_main_axis_fp
        sy = start_minor_axis_fp 

    # left top points 
    x_left = x2pixel(points[0][0], sx, img_w, axis=axis)
    y_top = y2pixel(points[0][1], sy, img_h, axis=axis)
    # right bottom points 
    x_right = x2pixel(points[1][0], sx, img_w, axis=axis)
    y_bottom = y2pixel(points[1][1], sy, img_h, axis=axis)
    return [[x_left, y_top], [x_right, y_bottom]]


def expanding_rect(points,
                   whole_img_h,
                   whole_img_w,
                   expand_ratio=1,
                   limit=False,
                   x_max=None,
                   y_max=None,
                   x_min=None,
                   y_min=None):
    """Expand or narrow rect size for labelme data"""

    if isinstance(expand_ratio, float) or isinstance(expand_ratio, int):
        x_ratio = y_ratio = expand_ratio
    elif isinstance(expand_ratio, list) or isinstance(expand_ratio, tuple):
        if len(expand_ratio) == 2:
            x_ratio, y_ratio = expand_ratio
        else:
            raise ValueError
    else:
        raise ValueError

    x, y, w, h = xyxy2xywh(points)
    expand_w = 0.5 * w * (1 + x_ratio)
    expand_h = 0.5 * h * (1 + y_ratio)

    if limit:
        if x_max:
            expand_w = min(expand_w, 0.5 * w + x_max)
        if x_min:
            expand_w = max(expand_w, 0.5 * w + x_min)
        if y_max:
            expand_h = min(expand_h, 0.5 * h + y_max)
        if y_min:
            expand_h = max(expand_h, 0.5 * h + y_min)

    x1 = max(x - expand_w, 0)
    y1 = max(y - expand_h, 0)
    x2 = min(x + expand_w, whole_img_w)
    y2 = min(y + expand_h, whole_img_h)

    x_left = min(x1, x2)
    x_right = max(x1, x2)
    y_top = min(y1, y2)
    y_bottom = max(y1, y2)

    return [[x_left, y_top], [x_right, y_bottom]]


def crop_segmented_rect(img, rect):
    """crop partial region from a 2D image"""
    img = img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
    return img

def pixel2fp(p, sp, length):
    return max(0, p / length + sp)

# frame2rect(points, start_main_axis_fp, img_h, img_w,  start_minor_axis_fp=0, axis=1)
def rect2frame(points, start_main_axis_px, img_h, img_w, start_minor_axis_px=0, axis=1):

    if axis == 0:
        sx = start_minor_axis_px
        sy = start_main_axis_px
    else:
        sx = start_main_axis_px
        sy = start_minor_axis_px 

    x_left = pixel2fp(points[0][0], sx, img_w)
    y_top = pixel2fp(points[0][1], sy, img_h)
    x_right = pixel2fp(points[1][0], sx, img_w)
    y_bottom = pixel2fp(points[1][1], sy, img_h)

    return [[x_left, y_top], [x_right, y_bottom]]

def xyxy_nested(curr_rect, reff_rect):
    pt1 = [abs(curr_rect[0][0] - reff_rect[0][0]), abs(curr_rect[0][1] - reff_rect[0][1])]
    pt2 = [abs(curr_rect[1][0] - reff_rect[0][0]), abs(curr_rect[1][1] - reff_rect[0][1])]
    return [pt1, pt2]


def ostu(s, th_start=0, th_end=256, th_step=1, return_hl=False):
    """OSTU algorithm"""
    max_sigma = 0
    max_th = 0

    for t in range(th_start, th_end, th_step):
        bg = s[s <= t]
        fg = s[s > t]

        p0 = bg.size / s.size
        p1 = fg.size / s.size

        m0 = 0 if bg.size == 0 else bg.mean()
        m1 = 0 if fg.size == 0 else fg.mean()

        sigma = p0 * p1 * (m0 - m1) * (m0 - m1)

        if sigma > max_sigma:
            max_sigma = sigma
            max_th = t

    if return_hl:
        bg = s[s <= max_th]
        fg = s[s > max_th]
        m0 = 0 if bg.size == 0 else bg.mean()
        m1 = 0 if fg.size == 0 else fg.mean()
        return int(max_th), m0, m1

    return int(max_th)
    
################################################################################
#### TODO later 
################################################################################


def pixel2x(x, sx, img_w):
    """pixel value (int) to frame values (float) in horizontal direction"""
    return x / img_w + sx


def pixel2y(y, sy, img_h):
    """pixel value (int) to frame values (float) in vertical direction"""
    return y / img_h + sy


# def rect2frame(points, sx, img_h, img_w, sy=0):
#     """left-top right bottom rect points to frames value (float)"""
#     x_left = pixel2x(points[0][0], sx, img_w)
#     y_top = pixel2y(points[0][1], sy, img_h)

#     x_right = pixel2x(points[1][0], sx, img_w)
#     y_bottom = pixel2y(points[1][1], sy, img_h)

#     return [(x_left, y_top), (x_right, y_bottom)]


def get_rect_area(points):
    """calculate area in a given rectangle"""
    w = abs(points[1][0] - points[0][0])
    h = abs(points[1][1] - points[0][1])
    return max(w * h, 0)


def imread_acc2_area(points, th1=0.003, th2=0.015):
    area = get_rect_area(points)
    if area < th1:
        imread = imread_full
    elif th1 <= area < th2:
        imread = imread_half
    else:
        imread = imread_quarter
    return imread


def read_segmented_rect(l, rect, imread):
    """Read rectangle"""
    img_h = imread(l[0]).shape[0]
    img = read_segmented_img(l, rect[0][0], rect[1][0], imread)
    y_top, y_bottom = y2pixel(rect[0][1], 0,
                              img_h), y2pixel(rect[1][1], 0, img_h)
    img = img[y_top:y_bottom, :]
    return img

# def imread_decorator(resize_ratio):
#     """Image read (GRAY)"""
#     def func(path, size=None, mode=cv2.IMREAD_GRAYSCALE, r=resize_ratio):
#         img = cv2.imread(path, mode)
#         if size is None:
#             img = cv2.resize(img, None, fx=r, fy=r)
#         else:
#             img = cv2.resize(img, size)
#         return img

#     func.resize_ratio = resize_ratio
#     return func


# def x2pixel(x, sx, img_w):
#     """frame values (float) to pixel (int) in horizontal direction"""
#     o = int(np.ceil((x - sx) * img_w))
#     o = max(o, 0)
#     return o


# def y2pixel(y, sy, img_h):
#     """frame values (float) to pixel (int) in vertical direction"""
#     o = int(np.ceil((y - sy) * img_h))
#     o = min(max(o, 0), img_h - 1)
#     return o


# def frame2rect(points, sx, img_h, img_w, sy=0):
#     """frame values (float) to pixel (int) in both horizontal and vertical direction"""
#     x_left = x2pixel(points[0][0], sx, img_w)
#     y_top = y2pixel(points[0][1], sy, img_h)

#     x_right = x2pixel(points[1][0], sx, img_w)
#     y_bottom = y2pixel(points[1][1], sy, img_h)
#     return [(x_left, y_top), (x_right, y_bottom)]
def trans_coords_from_chunk2frame(chunk_rect: list, item_rect: list):
    """translate item coords from ref-chunk to ref-frame

    Args:
        chunk_rect (list): chunk coords (ref-frame)
        item_rect (list): item coords (ref-chunk)

    Returns:
        _type_: item coords (ref-frame)
    """
    chunk_pt0, chunk_pt1 = chunk_rect
    X0, Y0 = chunk_pt0
    X1, Y1 = chunk_pt1
    chunk_h, chunk_w = Y1-Y0, X1-X0

    item_pt0, item_pt1 = item_rect
    x0, y0 = item_pt0
    x1, y1 = item_pt1

    new_x0 = (x0*chunk_w)+X0
    new_y0 = (y0*chunk_h)+Y0
    new_x1 = (x1*chunk_w)+X0
    new_y1 = (y1*chunk_h)+Y0
    return [[new_x0, new_y0], [new_x1, new_y1]]