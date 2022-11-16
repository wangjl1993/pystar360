import os
import zipfile
import numpy as np
from pathlib import Path


class imread3d_decorator:
    def __init__(self, h=None, w=None, inner_ext=".data"):
        self.inner_ext = inner_ext
        if h is not None and w is None:
            l = h
            self.reshape_image = lambda img, img_size: img.reshape((l, img_size // l))
        elif h is None and w is not None:
            l = w
            self.reshape_image = lambda img, img_size: img.reshape((img_size // l, l))
        elif h is not None and w is not None:
            self.reshape_image = lambda img, img_size: img.reshape((h, w))
        else:
            raise ValueError("Plase input .hx either height or width")

    def __call__(self, path):
        path = Path(path)
        with zipfile.ZipFile(str(path)) as z:
            f_name = path.stem + self.inner_ext
            with z.open(f_name, 'r') as f:
                img_byte = f.read()
                f.seek(0, os.SEEK_END)
                img_int = np.frombuffer(img_byte, np.uint16)
                img_size = len(img_int)
                img_int = self.reshape_image(img_int, img_size)
        return img_int


imread3d_h2000 = imread3d_decorator(h=2000)
imread3d_w1000 = imread3d_decorator(w=1000)


def remap_3d_to_2d(d_img, boundary_max=None, boundary_min=None, scale=255):
    # assert boundary_min >= 0, "boundary min must be greater than or equal to zero"
    # assert boundary_max > boundary_min, "Boundary max must be greater than or equal to boundary min"
    # when depth value == 0, that measn there is no valid depth value captured
    if boundary_max is None:
        boundary_max = np.max(d_img[d_img != 0])
        boundary_min = np.min(d_img[d_img != 0])
    scale_d_img = scale * (d_img.astype(np.float32) - boundary_min) / (boundary_max - boundary_min)
    scale_d_img[scale_d_img < 0] = 0
    scale_d_img[scale_d_img > 255] = 255
    return scale_d_img.astype(np.uint8)
