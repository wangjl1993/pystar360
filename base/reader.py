import re
import functools
from abc import ABCMeta, abstractclassmethod
from pathlib import Path

from pystar360.utilities.helper import get_img_size, imread_full

__all__ = ["ImReader"]


class ImReaderABC(metaclass=ABCMeta):
    @abstractclassmethod
    def _get_filters(self):
        raise NotImplementedError

    @abstractclassmethod
    def _check_file(self):
        raise NotImplementedError


@functools.lru_cache(maxsize=8)
class ImReader(ImReaderABC):
    def __init__(self,
                 images_path,
                 channel,
                 filter_ext=(".jpg"),
                 check_files=True,
                 logger=None,
                 debug=False,
                 verbose=False):

        self.images_path = images_path
        self.channel = channel
        self.filter_ext = filter_ext
        self.check_files = check_files
        self.logger = logger
        self.debug = debug

        self.filter_rules = self._get_filters()
        self._image_path_list = []

        self._read()

    def _read(self):
        p = Path(self.images_path)

        try:
            self._image_path_list = [i for i in p.iterdir() if i.suffix in self.filter_ext]
            if self.filter_rules:
                for myfilter in self.filter_rules:
                    self._image_path_list = myfilter(self._image_path_list)

            # sorted files in an acsending order
            self._image_path_list = sorted(self._image_path_list,
                                        key=lambda x:
                                        (int(re.sub("\D", "", x.name)), x))
            
        except Exception as e:
            if self.debug:
                print(e)
            raise FileNotFoundError(f">>> {str(self.images_path)}没有找到相对应的图像路径，请检查后路径是否正确") 
        
        if len(self._image_path_list) != 0:
            if self.check_files:
                self._check_file()
            
            self._image_path_list = list(map(str, self._image_path_list))
            if self.logger:
                self.logger.info(
                    f">>> The number of images listed in the given path: {self.__len__()}"
                )
            else:
                print(
                    f">>> The number of images listed in the given path: {self.__len__()}"
                )
        else:
            raise FileNotFoundError(f">>> The number of images listed in the given path: {self.__len__()}")

    def get_images_list(self):
        return self._image_path_list

    def __getitem__(self, index):
        return self._image_path_list[index]

    def __len__(self):
        return len(self._image_path_list)

    def _get_img_size(self, imread=imread_full):
        return get_img_size(self._image_path_list[0], imread)

    def _check_file(self):
        # check if there is a missing file
        pattern = list(map(lambda x: int(re.sub("\D", "", x.stem.split("-")[-1])), self._image_path_list))
        if pattern != list(range(pattern[0], pattern[-1] + 1, 1)):
            raise FileNotFoundError(f">>> File missing in {str(self.images_path)}!")

    def _get_filters(self):
        filter_rules = []
        filter_rule1 = lambda l: [
            p for p in l if p.stem.split("-")[0] == self.channel
        ]
        filter_rules.append(filter_rule1)
        return filter_rules