from time import time
from pystar360.utilities._logger import d_logger


class algoDecorator:
    def __init__(self, function, logger=None):
        self.function = function
        self.logger = logger

    def __call__(self, *args, **kwargs):
        t1 = time()
        result = self.function(*args, **kwargs)
        t2 = time()
        if self.logger:
            # self.logger.info(f'>>> Function {self.function.__name__!r} executed in {(t2-t1):.4f}s')
            self.logger.info(f'>>> Function executed in {(t2-t1):.6f}s')
        else:
            d_logger.info(f'>>> Function executed in {(t2-t1):.6f}s')
        return result


# def algoDecorator(func):
#     # This function shows the execution time of
#     # the function object passed
#     def wrap_func(*args, **kwargs):
#         t1 = time()
#         result = func(*args, **kwargs)
#         t2 = time()
#         print(f'>>> Function {func.__name__!r} executed in {(t2-t1):.4f}s')
#         return result
#     return wrap_func


class algoBaseABC:
    def __init__(self, **kwargs):
        for key, item in kwargs.items():
            setattr(self, key, item)


@algoDecorator
class NullDetection(algoBaseABC):
    def __call__(self, item_bboxes_list):
        # if empty, return empty
        if not item_bboxes_list:
            return []

        new_item_bboxes_list = []
        count = 1
        for _, box in enumerate(item_bboxes_list):
            box.index = count
            new_item_bboxes_list.append(box)
            count += 1

        return new_item_bboxes_list
