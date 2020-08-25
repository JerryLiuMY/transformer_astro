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


# global check
def check_dataset_name(dataset_name):
    assert dataset_name.split('_')[0] in ['ASAS', 'MACHO', 'WISE', 'GAIA', 'OGLE', 'Synthesis'], 'Invalid dataset name'


def check_model_name(model_name):
    assert model_name in ['sim', 'pha', 'att'], 'Invalid model name'


def check_set_type(set_type, is_fold=False):
    if not is_fold:
        assert set_type in ['train', 'valid', 'evalu'], 'Invalid set type'
    else:
        assert set_type in ['train', 'evalu'], 'Invalid set type'

