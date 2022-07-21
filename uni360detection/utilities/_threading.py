import threading


def threadingDecorator(func):

    def wrap_func(*args, **kwargs):
        th = threading.Thread(target=func,
                              args=args,
                              kwargs=kwargs)
        th.start()
        return 
    
    return wrap_func 