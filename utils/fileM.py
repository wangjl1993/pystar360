import re
import json
from pathlib import Path
from filelock import FileLock  # mutex lock

IMAGE_EXT = [".png", ".jpg", "jpeg", ".bmp", ".tif"]


def list_images_in_path(path,
                        filter_rules=[],
                        filter_ext=[".png", ".jpg", "jpeg"],
                        verbose=True,
                        check_missing=True,
                        logger=None):
    """list images in a given path"""
    # list file in path with image extention
    p = Path(str(path))
    images_l = [i for i in p.iterdir() if i.suffix in filter_ext]

    # filter rules
    if filter_rules:
        for filter in filter_rules:
            images_l = filter(images_l)

    # sorted files in an acsending order
    images_l = sorted(images_l,
                      key=lambda x: (int(re.sub("\D", "", x.name)), x))

    # check if there is a missing file
    if check_missing:
        pattern = list(map(lambda x: int(re.sub("\D", "", x.name)), images_l))
        if pattern != list(range(pattern[0], pattern[-1] + 1, 1)):
            raise ValueError(f">>> File missing in {str(p)}!")

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
