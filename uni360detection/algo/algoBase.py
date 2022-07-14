
from time import time


class algoDecorator:
    def __init__(self, function, logger=None):
        self.function = function 
        self.logger = logger 

    def __call__(self, *args, **kwargs):
        t1 = time()
        result = self.function(*args, **kwargs)
        t2 = time()
        if self.logger:
            self.logger.info(f'>>> Function {self.function.__name__!r} executed in {(t2-t1):.4f}s')
        else:
            print(f'>>> Function {self.function.__name__!r} executed in {(t2-t1):.4f}s')
        return result 

class algoBase:
    def __init__(self, **kwargs):
        for key, item in kwargs.items():
            setattr(self, key, item)  
            
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
