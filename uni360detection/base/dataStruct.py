from dataclasses import dataclass, field, fields, is_dataclass
from typing import List, Optional, Tuple, Union
from uni360detection.utilities.helper import get_label_num2check

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
    label: str = "" # xxx#2 比如xxls=3 就是name=‘xxls’，num2check=3，至少需要检查3个xxls；或xxls 就是name=‘xxls’，num2check（default=1，默认至少检测1个
    index: int = 0 # order, 在这辆车，从左到右，或者从上到下，第几个这样的item
    name: str = "" # xxx item name label的名字
    num2check: int = 1 # # number of item to check # 需要检测的个数
    _orig_rect: Union[Rect, List, Tuple] = field(default_factory=Rect) # optional， 原目标物多大
    _temp_rect: Union[Rect, List, Tuple] = field(default_factory=Rect) # rect position in template, 检测框（映射直接用的框）; 比如用yolo的话，就需要把orig_rect扩大一定大小；不需要的话，就是目标物框
    _proposal_rect: Union[Rect, List, Tuple] = field(default_factory=Rect) # proposal rect position，初步映射到测试图的框，
    _curr_rect: Union[Rect, List, Tuple] = field(default_factory=Rect) # rect position in current train 如果有进一步检测，返回yolo精确框；不需要的话，等于proposal_rect
    _hist_rect: Union[Rect, List, Tuple] = field(default_factory=Rect) # if needed, 预留历史图的框，一般不需要，optional
    conf_score: float = 0. # confidence level 无论什么方法，计算出来的置信度
    conf_thres: float = 0. # confidence threshold 置信度的评判阈值是多少
    is_defect: int = 0. # if it is defect, 0 is ok, 1 is ng 是否故障
    # optional 
    value: Union[float, List] = 0. # for a measurement method 如果使用度量方法，测试的数值是多少
    value_thres: float = 0. # 度量的阈值是多少
    unit: str = "" # unit, like mm, cm if needed  # 单位是多少
    defect_type: int = 0 # defect type if needed # 故障类型是什么，预留
    description: str = "" # defect description # 故障说明，可以写或者不写 预留
    is_3ddefect: int = 0
    value_3d: float = 0
    value_3dthres: Union[float, List] = 0. 

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
    def orig_rect(self):
        return self._orig_rect
    
    @orig_rect.setter
    def orig_rect(self, value):
        self._orig_rect = Rect(*value)

    @property 
    def hist_rect(self):
        return self._hist_rect
    
    @hist_rect.setter
    def hist_rect(self, value):
        self._hist_rect = Rect(*value)
    
    def __validate(self):
        for f in fields(self):
            if f.name in ("temp_rect", "curr_rect", "proposal_rect", "orig_rect", "hist_rect"):
                value = getattr(self, f.name)
                assert len(value) == 2
                setattr(self, f.name, Rect(*value))

def bbox_formater(bboxes):
    """convert to data struct"""
    # bboxes: dict
    bboxes = sorted(bboxes, key=lambda x: x["temp_rect"][0][0])
    
    new_bboxes = []
    for b in bboxes:
        name, num2check = get_label_num2check(b["label"])
        box = BBox(label=b["label"], name=name, num2check=num2check)
        box.orig_rect = b["orig_rect"]
        box.temp_rect = b["temp_rect"]
        new_bboxes.append(box)
    
    return new_bboxes

@dataclass
class CarriageInfo:
    path: str = ""
    startline: float = 0.
    endline: float = 0.
    first_axis_idx: int = 0
    second_axis_idx: int = 0

@dataclass
class QTrainInfo:
    major_train_code: str = "" # CRH1A
    minor_train_code: str = "" # CRH1A-A
    train_num: str= "" # 1178, CRH1A-A 1178
    train_sn: str = "" # 2101300005, date or uid 
    channel: str = "" # 12,4,17...
    carriage: int = 0 # 1-8 or 1-16
    test_train: CarriageInfo = field(default_factory=CarriageInfo)
    hist_train: Optional[CarriageInfo] = field(default_factory=CarriageInfo)
    direction: Optional[int] = 0
    Pantograph_state: int = 0