from abc import ABCMeta, abstractclassmethod


class BaseInfer(metaclass=ABCMeta):
    @abstractclassmethod
    def _initialize(self):
        raise NotImplementedError

    @abstractclassmethod
    def infer(self, img):
        raise NotImplementedError
