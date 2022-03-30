from dataclasses import dataclass


@dataclass
class BBox:
    label: str = ""
    number: str = ""
    ref_points: list = [[0, 0], [0, 0]]
    cur_points: list = [[0, 0], [0, 0]]
    proposal_rect: list = [[0, 0], [0, 0]]
    proposal_region: list = [[0, 0], [0, 0]]
    score: float = 0
    is_error: int = 0