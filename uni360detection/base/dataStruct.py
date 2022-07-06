from dataclasses import dataclass,  fields, field, is_dataclass
from typing import List, Optional, Union, Tuple


@dataclass
class Point(List):
    x: Union[int, float] = 0.
    y: Union[int, float] = 0.

    def __post_init__(self):
        self.__field_name = tuple(f.name for f in fields(self))
        assert len(self.__field_name) == 2

    def __getitem__(self, index):
        return getattr(self, self.__field_name[index], None)
        
    def __setitem__(self, index, value):
        setattr(self, self.__field_name[index], value)

    def __len__(self):
        return len(self.__field_name)

    def to_tuple(self):
        return tuple(getattr(self, field.name).to_tuple() \
            if is_dataclass(getattr(self, field.name)) else getattr(self, field.name) \
             for field in fields(self))

    def to_list(self):
        return list(getattr(self, field.name).to_list() \
            if is_dataclass(getattr(self, field.name)) else getattr(self, field.name) \
                for field in fields(self))

@dataclass 
class Rect(List):
    _pt1: Union[List, Tuple, Point] = field(default_factory=Point)
    _pt2: Union[List, Tuple, Point] = field(default_factory=Point)

    def __post_init__(self):
        self.__validate()
        self.__field_name = tuple(f.name for f in fields(self))
        assert len(self.__field_name) == 2

    def __getitem__(self, index):
        return getattr(self, self.__field_name[index], None)
        
    def __setitem__(self, index, value):
        if isinstance(value, List) or isinstance(value, Tuple) or isinstance(value, Point):
            value = Point(*value)
        setattr(self, self.__field_name[index], value)
    
    def __len__(self):
        return len(self.__field_name)

    def __validate(self):
        for f in fields(self):
            if not isinstance(getattr(self, f.name), Point):
                value = getattr(self, f.name)
                assert len(value) == 2
                setattr(self, f.name, Point(*value))
    @property 
    def pt1(self):
        return self._pt1
    
    @pt1.setter
    def pt1(self, value):
        self._pt1 = Point(*value)

    @property 
    def pt2(self):
        return self._pt2
    
    @pt2.setter
    def pt2(self, value):
        self._pt2 = Point(*value)

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

    def to_list(self):
        return list(getattr(self, field.name).to_list() \
            if is_dataclass(getattr(self, field.name)) else getattr(self, field.name) \
                for field in fields(self))

@dataclass
class BBox:
    label: str = "" # xxx-#
    index: int = 0 # 
    name: str = "" # xxx item name 
    num2check: int = 0 # # number of item to check 
    _temp_rect: Union[Rect, List, Tuple] = field(default_factory=Rect)
    _curr_rect: Union[Rect, List, Tuple] = field(default_factory=Rect)
    _proposal_region: Union[Rect, List, Tuple] = field(default_factory=Rect)
    score: float = 0. # confidence level 
    conf_thres: float = 0. # confidence threshold 
    is_defect: int = 0. # if it is defect, 0 is ok, 1 is ng
    # optional 
    value: float = 0. # for a measurement method 
    unit: str = "" # unit, like mm, cm if needed 
    defect_type: int = 0 # defect type if needed 
    description: str = "" # defect description 
    _proposal_rect: Union[Rect, List, Tuple] = field(default_factory=Rect) # if needed 
    _hist_rect: Union[Rect, List, Tuple] = field(default_factory=Rect) # if needed 

    def __post_init__(self):
        self.__validate()
    
    @property 
    def temp_rect(self):
        return self._temp_rect
    
    @temp_rect.setter
    def temp_rect(self, value):
        self._temp_rect = Rect(*value)

    @property 
    def curr_rect(self):
        return self._curr_rect
    
    @curr_rect.setter
    def curr_rect(self, value):
        self._curr_rect = Rect(*value)

    @property 
    def proposal_rect(self):
        return self._proposal_rect
    
    @proposal_rect.setter
    def proposal_rect(self, value):
        self._proposal_rect = Rect(*value)

    @property 
    def proposal_region(self):
        return self._proposal_region
    
    @proposal_region.setter
    def proposal_region(self, value):
        self._proposal_region = Rect(*value)

    @property 
    def hist_rect(self):
        return self._hist_rect
    
    @hist_rect.setter
    def hist_rect(self, value):
        self._hist_rect = Rect(*value)
    
    def __validate(self):
        for f in fields(self):
            if f.name in ("temp_rect", "curr_rect", "proposal_rect", "proposal_region", "hist_rect"):
                value = getattr(self, f.name)
                assert len(value) == 2
                setattr(self, f.name, Rect(*value))

@dataclass
class CarriageInfo:
    path: str = ""
    startline: float = 0.
    endline: float = 0.

@dataclass
class QTrainInfo:
    major_train_code: str = "" # CRH1A
    minor_train_code: str = "" # CRH1A-A
    train_num: str= "" # 1178, CRH1A-A 1178
    train_sn: str = "" # 2101300005, date or uid 
    channel: str = "" # 12,4,17...
    carriage: int = 0 # 1-8
    test_train: CarriageInfo = field(default_factory=CarriageInfo)
    hist_train: Optional[CarriageInfo] = field(default_factory=CarriageInfo)
    direction: Optional[int] = 0