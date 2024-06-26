from pystar360.utilities.deviceController import time_sync
from pystar360.utilities.logger import w_logger


class algoDecorator:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        t1 = time_sync()
        result = self.function(*args, **kwargs)
        t2 = time_sync()
        w_logger.info(f'>>> Function executed in {(t2-t1):.6f}s')
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

    def __call__(self, *args, **kwargs):
        pass


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
