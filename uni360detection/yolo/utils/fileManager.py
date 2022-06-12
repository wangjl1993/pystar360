import re
import os
import json
from pathlib import Path
from filelock import FileLock  # mutex lock

IMAGE_EXT = [".png", ".jpg", "jpeg", ".bmp", ".tif"]


def list_images_in_path(path,
                        filter_rules=[],
                        filter_ext=[".jpg"],
                        verbose=True,
                        logger=None):
    """list images in a given path"""
    # list file in path with image extention
    p = Path(str(path))
    images_l = [i for i in p.iterdir() if i.suffix in filter_ext]

    # filter rules
    if filter_rules:
        for myfilter in filter_rules:
            images_l = myfilter(images_l)

    # sorted files in an acsending order
    images_l = sorted(images_l,
                      key=lambda x: (int(re.sub("\D", "", x.name)), x))

    # console output
    if verbose:
        if logger:
            logger.info(
                f">>> The number of images listed in the given path: {len(images_l)}"
            )
        else:
            print(
                f">>> The number of images listed in the given path: {len(images_l)}"
            )

    images_l = list(map(str, images_l))
    return images_l


def write_json(path, json_dict, lock=False, mode="w", indent=2):
    """Write json"""
    if lock:
        # if mutex lock
        with FileLock(path + ".lock"):
            with open(path, mode) as f:
                json.dump(json_dict, f, indent=indent, ensure_ascii=False)
    else:
        with open(path, mode) as f:
            json.dump(json_dict, f, indent=indent, ensure_ascii=False)
    print(f">>> Generate {path}...")


def read_json(path, mode="rb"):
    """Read json"""
    with open(path, mode) as f:
        load_dict = json.load(f)
    return load_dict


def mkdirs(save_path, sub_folder=None):
    """Make folders"""
    if sub_folder is not None:
        new_path = os.path.join(save_path, sub_folder)
    else:
        new_path = save_path
    os.makedirs(new_path, exist_ok=True)
    return new_path