import numpy as np 
from dataclasses import dataclass, fields, field, astuple, is_dataclass
from typing import List, Optional, Union, Tuple


@dataclass
class Point(List):
    x: Union[int, float] = 0.
    y: Union[int, float] = 0.

    def __post_init__(self):
        self.__field_name = tuple(f.name for f in fields(self))
        assert len(self.__field_name) <= 2

    def __getitem__(self, index):
        return getattr(self, self.__field_name[index], None)
        
    def __setitem__(self, index, value):
        setattr(self, self.__field_name[index], value)

    def to_tuple(self):
        return tuple(getattr(self, field.name).to_tuple() \
            if is_dataclass(getattr(self, field.name)) else getattr(self, field.name) \
             for field in fields(self))

@dataclass 
class Rect():
    pt1: Union[List, Tuple, Point] = field(default_factory=Point)
    pt2: Union[List, Tuple, Point] = field(default_factory=Point)

    def __post_init__(self):
        self.__field_name = tuple(f.name for f in fields(self))
        assert len(self.__field_name) <= 2

    def __getitem__(self, index):
        return getattr(self, self.__field_name[index], None)
        
    def __setitem__(self, index, value):
        setattr(self, self.__field_name[index], value)

    def get_size(self):
        h = abs(self.pt2[1] - self.pt1[0])
        w = abs(self.pt2[0] - self.pt1[0])
        return h, w 

    def get_area(self):
        h, w = self.get_size()
        return h*w

    def get_ctr(self):
        h, w = self.get_size()
        ctrx = self.pt1[0] + 0.5 * w
        ctry = self.pt1[0] + 0.5 * h 
        return ctrx, ctry 

    def to_tuple(self):
        return tuple(getattr(self, field.name).to_tuple() \
            if is_dataclass(getattr(self, field.name)) else getattr(self, field.name) \
                for field in fields(self))

def initialize_rect():
    return [Point() for _ in range(2)]

@dataclass
class BBox:
    label: str = "" # xxx-#
    name: str = "" # xxx item name 
    num2check: int = 0 # # number of item to check 
    temp_rect: Union[Rect, List, Tuple] = field(default_factory=Rect)
    curr_rect: Union[Rect, List, Tuple] = field(default_factory=Rect)
    proposal_rect: Union[Rect, List, Tuple] = field(default_factory=Rect)
    proposal_region: Union[Rect, List, Tuple] = field(default_factory=Rect)
    score: float = 0
    is_defect: int = 0


# @dataclass
# class Carriage:
#     path: str = ""
#     startline: float = 0.
#     endline: float = 0.
#     img: np.array = np.array([])

# @dataclass
# class QTrain_info:
#     train_sn: str = ""
#     major_train_code: str = ""
#     minor_train_code: str = ""
#     channel: str = ""
#     carriage: int = 0
#     test_train: Carriage = Carriage("", 0, 0, np.array([]))
#     hist_train: Optional[Carriage] = Carriage("", 0, 0, np.array([]))