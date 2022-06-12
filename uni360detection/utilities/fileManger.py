import re
import os
import json

from omegaconf import OmegaConf
from pathlib import Path
from filelock import FileLock  # mutex lock


def sort_files_in_numeric_order(l):
    """Sort fiels in number order"""
    return sorted(l, key=lambda x: (int(re.sub("\D", "", x)), x))


def mkdirs(save_path, sub_folder=None):
    """Make folders"""
    if sub_folder is not None:
        new_path = os.path.join(save_path, sub_folder)
    else:
        new_path = save_path
    os.makedirs(new_path, exist_ok=True)
    return new_path


def write_json(path, dict, lock=False, mode="w"):
    """Write json"""
    if lock:
        with FileLock(path + ".lock"):
            with open(path, mode) as f:
                json.dump(dict, f, indent=4, ensure_ascii=False)
    else:
        with open(path, mode) as f:
            json.dump(dict, f, indent=4, ensure_ascii=False)
    print(f">>> Generate {path}...")


def read_json(path, mode="rb"):
    """Read json"""
    with open(path, mode) as f:
        load_dict = json.load(f)
    return load_dict

def read_yaml(path):
    return OmegaConf.load(path)

def write_yaml(path, dict, lock=False):
    if lock:
        with FileLock(path + ".lock"):
            OmegaConf.save(config=dict, f=path)
    else:
        OmegaConf.save(config=dict, f=path)
    print(f">>> Generate {path}...")


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