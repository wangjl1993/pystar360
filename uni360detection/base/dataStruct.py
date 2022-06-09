from dataclasses import dataclass, field
from typing import List


def initialize_rect():
    return [[0, 0] for _ in range(2)]


@dataclass
class BBox:
    label: str = ""
    number: int = 0
    ref_rect: list = field(default_factory=initialize_rect)
    cur_rect: list = field(default_factory=initialize_rect)
    proposal_rect: list = field(default_factory=initialize_rect)
    proposal_region: list = field(default_factory=initialize_rect)
    score: float = 0
    is_defect: int = 0


@dataclass
class train_info:
    train_sn: str = ""
    major_train_code: str = ""
    minor_train_code: str = ""
    channel: str = ""
