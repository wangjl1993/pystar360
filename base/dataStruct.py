from dataclasses import dataclass, field, fields, is_dataclass
from dataclasses_json import dataclass_json
from typing import (
    List, Optional, Tuple, 
    Union, Any, Callable
)
from pystar360.utilities.helper import get_label_num2check
import numpy as np
import copy
from pystar360.utilities.logger import w_logger, set_task_id

@dataclass_json
@dataclass
class Point(List):
    x: Union[int, float] = 0.0
    y: Union[int, float] = 0.0

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
        return tuple(
            getattr(self, field.name).to_tuple()
            if is_dataclass(getattr(self, field.name))
            else getattr(self, field.name)
            for field in fields(self)
        )

    def to_list(self):
        return list(
            getattr(self, field.name).to_list()
            if is_dataclass(getattr(self, field.name))
            else getattr(self, field.name)
            for field in fields(self)
        )


@dataclass_json
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
        return h * w

    def get_ctr(self):
        h, w = self.get_size()
        ctrx = self.pt1[0] + 0.5 * w
        ctry = self.pt1[0] + 0.5 * h
        return ctrx, ctry

    def to_tuple(self):
        return tuple(
            getattr(self, field.name).to_tuple()
            if is_dataclass(getattr(self, field.name))
            else getattr(self, field.name)
            for field in fields(self)
        )

    def to_list(self):
        return list(
            getattr(self, field.name).to_list()
            if is_dataclass(getattr(self, field.name))
            else getattr(self, field.name)
            for field in fields(self)
        )
    
    def is_none(self, ):
        if self.pt1 == Point() and self.pt2 == Point():
            return True 
        else:
            return False

    def map_to_rect3d(self, axis: int,  map3d_minor_axis_poly_func: Callable, map3d_major_axis_poly_func: Callable):
        if self.is_none():
            return Rect()
        if axis:
            new_rect = Rect(
                [map3d_major_axis_poly_func(self.pt1.x), map3d_minor_axis_poly_func(self.pt1.y)],
                [map3d_major_axis_poly_func(self.pt2.x), map3d_minor_axis_poly_func(self.pt2.y)]
            )
        else:
            new_rect = Rect(
                [map3d_minor_axis_poly_func(self.pt1.x), map3d_major_axis_poly_func(self.pt1.y)],
                [map3d_minor_axis_poly_func(self.pt2.x), map3d_major_axis_poly_func(self.pt2.y)]
            )
        return new_rect


@dataclass_json
@dataclass
class BBox:
    ############ 基本信息 ############
    # xxx#2 比如xxls=3 就是name=‘xxls’，num2check=3，至少需要检查3个xxls；或xxls 就是name=‘xxls’，num2check（default=1），默认至少检测1个
    label: str = ""
    # order, 在这辆车，从左到右，或者从上到下，第几个这样的item
    index: int = 0
    # xxx item name label的名字
    name: str = ""
    # number of item to check # 需要检测的个数
    num2check: int = 1

    ############  ############
    # 目标物多大 目标物品的bounding box(2d)
    _orig_rect: Union[Rect, List, Tuple] = field(default_factory=Rect)
    # rect position in template, 检测框（映射直接用的框）; 比如用yolo的话，就需要把orig_rect扩大一定大小变成temp_rect；不需要的话，就是目标物框(2d)
    _temp_rect: Union[Rect, List, Tuple] = field(default_factory=Rect)
    # proposal rect position，初步映射到测试图的框(2d)
    _curr_proposal_rect: Union[Rect, List, Tuple] = field(default_factory=Rect)
    # rect position in current train 如果有进一步检测，返回yolo精确框；不需要的话，等于proposal_rect (2d)
    _curr_rect: Union[Rect, List, Tuple] = field(default_factory=Rect)
    # proposal rect position, 如果3d图像单独定位，就直接用 curr_proposal_rect3d, 如果3d定位是从2d配准得到的就是用porposal rect
    _curr_proposal_rect3d: Union[Rect, List, Tuple] = field(default_factory=Rect)
    # 3d 目标的bounding bbox
    _curr_rect3d: Union[Rect, List, Tuple] = field(default_factory=Rect)
    # 【optional预留】，# if needed, 预留历史图的框，一般不需要，optional
    _hist_rect: Union[Rect, List, Tuple] = field(default_factory=Rect)
    # 【optional预留】，# if needed, 预留历史图3d的框，一般不需要，optional
    _hist_rect3d: Union[Rect, List, Tuple] = field(default_factory=Rect)
    _hist_proposal_rect: Union[Rect, List, Tuple] = field(default_factory=Rect)
    _hist_proposal_rect3d: Union[Rect, List, Tuple] = field(default_factory=Rect)

    ############ 2d图像评价 ############
    # confidence level 无论什么方法，计算出来的置信度，信心程度
    conf_score: float = 0.0
    # confidence threshold 置信度的评判阈值是多少
    conf_thres: float = 0.0
    # if it is defect, 0 is ok, 1 is ng 是否故障
    is_defect: int = 0

    ############ 3d图像评价 ############
    # 3d 深度数值
    value_3d: Union[float, List] = 0.0
    # 3d 深度阈值
    value_3dthres: Union[float, List] = 0.0
    # defect description # 故障说明，可以写或者不写
    description: str = ""
    # 3d 是否出现故障
    is_3ddefect: int = 0

    ############ 测量值评价 ############
    # for a measurement method 如果使用度量方法，测试的数值是多少【像素】
    value: Union[float, List] = 0.0
    # 度量的阈值是多少 optional
    value_thres: float = 0.0
    # unit, like mm, cm if needed 单位是什么 【预留 TODO】
    unit: str = ""

    ############ 其他 ############
    # defect type if needed # 故障类型是什么，预留设计 TODO
    defect_type: int = 0
    # 是否使用算法检测过
    is_detected: int = 0  # 是否被算法检查过
    # 使用多少次算法
    algo_cnt: int = 0

    def __post_init__(self):
        self.__validate()

    @property
    def temp_rect(self):
        return self._temp_rect

    @temp_rect.setter
    def temp_rect(self, value):
        if isinstance(value, Rect):
            self._temp_rect = value
        else:
            self._temp_rect = Rect(*value)

    @property
    def curr_rect(self):
        return self._curr_rect

    @curr_rect.setter
    def curr_rect(self, value):
        if isinstance(value, Rect):
            self._curr_rect = value
        else:
            self._curr_rect = Rect(*value)

    @property
    def curr_proposal_rect(self):
        return self._curr_proposal_rect

    @curr_proposal_rect.setter
    def curr_proposal_rect(self, value):
        if isinstance(value, Rect):
            self._curr_proposal_rect = value
        else:
            self._curr_proposal_rect = Rect(*value)

    @property
    def curr_proposal_rect3d(self):
        return self._curr_proposal_rect3d

    @curr_proposal_rect3d.setter
    def curr_proposal_rect3d(self, value):
        if isinstance(value, Rect):
            self._curr_proposal_rect3d = value
        else:
            self._curr_proposal_rect3d = Rect(*value)

    @property
    def curr_rect3d(self):
        return self._curr_rect3d

    @curr_rect3d.setter
    def curr_rect3d(self, value):
        if isinstance(value, Rect):
            self._curr_rect3d = value
        else:
            self._curr_rect3d = Rect(*value)

    @property
    def orig_rect(self):
        return self._orig_rect

    @orig_rect.setter
    def orig_rect(self, value):
        if isinstance(value, Rect):
            self._orig_rect = value
        else:
            self._orig_rect = Rect(*value)

    @property
    def hist_rect(self):
        return self._hist_rect

    @hist_rect.setter
    def hist_rect(self, value):
        if isinstance(value, Rect):
            self._hist_rect = value
        else:
            self._hist_rect = Rect(*value)

    @property
    def hist_rect3d(self):
        return self._hist_rect

    @hist_rect3d.setter
    def hist_rect3d(self, value):
        if isinstance(value, Rect):
            self._hist_rect3d = value
        else:
            self._hist_rect3d = Rect(*value)

    @property
    def hist_proposal_rect(self):
        return self._hist_proposal_rect

    @hist_proposal_rect.setter
    def hist_proposal_rect(self, value):
        if isinstance(value, Rect):
            self._hist_proposal_rect = value
        else:
            self._hist_proposal_rect = Rect(*value)

    @property
    def hist_proposal_rect3d(self):
        return self._hist_proposal_rect3d

    @hist_proposal_rect3d.setter
    def hist_proposal_rect3d(self, value):
        if isinstance(value, Rect):
            self._hist_proposal_rect3d = value
        else:
            self._hist_proposal_rect3d = Rect(*value)

    def __validate(self):
        for f in fields(self):
            if f.name in ("temp_rect", "orig_rect", "curr_rect", "curr_proposal_rect",  "hist_rect", "hist_proposal_rect"):
                value = getattr(self, f.name)
                assert len(value) == 2
                setattr(self, f.name, Rect(*value))


def json2bbox_formater(bboxes):
    """convert to data struct"""
    if not bboxes:
        return []

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


def bboxes_collector(bboxes):
    """collect items according to its label"""
    output = {}
    for b in bboxes:
        if b.name in output:
            output[b.name].append(b)
        else:
            output[b.name] = [b]
    return output


@dataclass_json
@dataclass
class CarriageInfo:
    path: str = ""  # 图像文件地址
    img_h: Optional[int] = 0 # 一帧图片 h
    img_w: Optional[int] = 0 # 一帧图片 w
    startline: float = 0.0  # 车体起始位置
    endline: float = 0.0  # 车体结束位置
    img: Optional[np.ndarray] = None # 整节车厢拼接图
    anchors: Optional[List[BBox]] = field(default_factory=list)
    normal_items: Optional[List[BBox]] = field(default_factory=list)
    static_chunks: Optional[List[Tuple[str, List[BBox]]]] = field(default_factory=list)
    dynamic_chunks: Optional[List] = field(default_factory=list)


    __fronze_fields__ = {
        "startline": 1, "endline": 1, "img_h": 1, "img_w": 1, "path": 1, "img": 1,
        "anchors": 1, "normal_items": 1, "static_chunks": 1, "dynamic_chunks": 1
    } 
    __fronze_fields_cnt__ = {
        "startline": 0, "endline": 0, "img_h": 0, "img_w": 0, "path": 0, "img": 0,
        "anchors": 0, "normal_items": 0, "static_chunks": 0, "dynamic_chunks": 0
    }    

    def __setattr__(self, __name: str, __value: Any) -> None:
        # print("set: ", __name)
        self_count_key = f"__{id(self)}_fronze_fields_cnt__"
        try:
            self_count_value = getattr(self, self_count_key)
        except AttributeError:
            _val = copy.deepcopy(self.__fronze_fields_cnt__)
            super().__setattr__(self_count_key, _val)
            self_count_value = _val
        
        cur_val = self_count_value.get(__name, 0)
        if cur_val > self.__fronze_fields__[__name]:
            raise ValueError("CarriageInfo attri cannot be set again.")
        super().__setattr__(__name, __value)
        self_count_value[__name] = cur_val+1



@dataclass_json
@dataclass
class QTrainInfo:
    major_train_code: str = ""  # 型号 CRH1A
    minor_train_code: str = ""  # 子型号 CRH1A-A
    train_num: str = ""  # 车号 1178, CRH1A-A 1178
    train_sn: str = ""  # 运行编号 2101300005, date or uid
    channel: str = ""  # 相机通道 12,4,17...
    carriage: int = 0  # 辆位 1-8 or 1-16
    curr_train2d: Optional[CarriageInfo] = None  # 测试车
    curr_train3d: Optional[CarriageInfo] = None  # 测试车
    hist_train2d: Optional[CarriageInfo] = None  # 历史车
    hist_train3d: Optional[CarriageInfo] = None  # 历史车
    item_bboxes: Optional[List[BBox]] = field(default_factory=list) # 所有检测项点收集到这里
    temp_train: Optional[CarriageInfo] = field(default_factory=CarriageInfo)  # 模版车
    direction: Optional[int] = 0  # 方向
    Pantograph_state: int = 0  # 升降弓
    is_concat: int = 0  # 是否是重联组，0不是重联组，1是重联组
    img2d_ext: str = ".jpg"  # 2d图像后缀
    img3d_ext: str = ".data"  # 深度图后缀

    def __post_init__(self,):
        # 任务id
        self.unique_repr_of_task = f"{self.minor_train_code}_{self.train_num}_{self.channel}_{self.carriage}_{self.train_sn}"
        set_task_id(self.unique_repr_of_task)
        if self.curr_train2d is not None:
            w_logger.info(f"curr_train2d path: {self.curr_train2d.path}")
        if self.curr_train3d is not None:
            w_logger.info(f"curr_train3d path: {self.curr_train3d.path}")
        if self.hist_train2d is not None:
            w_logger.info(f"hist_train2d path: {self.hist_train2d.path}")
        if self.hist_train3d is not None:
            w_logger.info(f"hist_train3d path: {self.hist_train3d.path}")

        w_logger.info(f'direction: {self.direction}')
        

    # def __getattribute__(self, __name: str) -> Any:
    #     print("get: ", __name)
    #     return super().__getattr__(__name)
    
    # def __setattr__(self, __name: str, __value: Any) -> None:
    #     print("set: ", __name)
    #     super().__setattr__(__name, __value)