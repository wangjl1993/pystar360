import numpy as np


def concat_str(*args, separator="_"):
    """Concatenate string"""
    args = list(map(str, args))
    return separator.join(args)