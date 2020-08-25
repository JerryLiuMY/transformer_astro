import functools
import time


# data tools
def one_hot_msg(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = func(*args, **kwargs)
        print(f'Successfully loaded one-hot encoder')
        return value

    return wrapper


def data_msg(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        beg_time = time.process_time()
        value = func(*args, **kwargs)
        end_time = time.process_time()
        print(f'Successfully loaded pre-processed data in {round(end_time - beg_time, 2)}s')
        return value

    return wrapper


