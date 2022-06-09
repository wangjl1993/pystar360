import re

from pathlib import Path
from abc import ABCMeta, abstractclassmethod

__all__ = ["ImReader"]


class ImReaderABC(metaclass=ABCMeta):
    @abstractclassmethod
    def _get_filters(self):
        raise NotImplementedError

    @abstractclassmethod
    def _check_file(self):
        raise NotImplementedError


class ImReader(ImReaderABC):
    def __init__(self,
                 images_path,
                 channel,
                 filter_ext=[".jpg"],
                 check_files=True,
                 verbose=False,
                 logger=None):

        self.images_path = images_path
        self.channel = channel
        self.filter_ext = filter_ext
        self.check_files = check_files
        self.verbose = verbose
        self.logger = logger
        self.filter_rules = self._get_filters()
        self._image_path_list = []

    def read(self):
        p = Path(self.images_path)
        self._image_path_list = [
            i for i in p.iterdir() if i.suffix in self.filter_ext
        ]

        if self.filter_rules:
            for myfilter in self.filter_rules:
                self._image_path_list = myfilter(self._image_path_list)

        # sorted files in an acsending order
        self._image_path_list = sorted(self._image_path_list,
                                       key=lambda x:
                                       (int(re.sub("\D", "", x.name)), x))

        if self.check_files:
            self._check_file()

        # console output
        if self.verbose:
            if self.logger:
                self.logger.info(
                    f">>> The number of images listed in the given path: {self.__len__()}"
                )
            else:
                print(
                    f">>> The number of images listed in the given path: {self.__len__()}"
                )

        self._image_path_list = list(map(str, self._image_path_list))

    def _check_file(self):
        # check if there is a missing file
        pattern = list(
            map(lambda x: int(re.sub("\D", "",
                                     x.stem.split("-")[-1])),
                self._image_path_list))
        if pattern != list(range(pattern[0], pattern[-1] + 1, 1)):
            raise ValueError(f">>> File missing in {str(self.images_path)}!")

    def _get_filters(self):
        filter_rules = []
        filter_rule1 = lambda l: [
            p for p in l if p.stem.split("-")[0] == self.channel
        ]
        filter_rules.append(filter_rule1)
        return filter_rules

    def get_images_list(self):
        return self._image_path_list

    def __getitem__(self, index):
        return self._image_path_list[index]

    def __len__(self):
        return len(self._image_path_list)