import functools
import os
import time
import numpy as np


def new_dir(log_dir):
    past_dirs = next(os.walk(log_dir))[1]
    new_num = 0 if len(past_dirs) == 0 else np.max([int(past_dir.split('_')[-1]) for past_dir in past_dirs]) + 1
    exp_dir = os.path.join(log_dir, '_'.join(['experiment', str(new_num)]))
    os.mkdir(exp_dir)

    return exp_dir


def timer(func):
    print('Loading pre-processed data ...')

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        beg_time = time.process_time()
        value = func(*args, **kwargs)
        end_time = time.process_time()
        print(f'Successfully loaded pre-processed data in {round(end_time - beg_time, 2)}s')
        return value

    return wrapper


def check_dataset_name(dataset_name):
    assert dataset_name.split('_')[0] in ['ASAS', 'MACHO', 'WISE', 'GAIA', 'OGLE', 'Synthesis'], 'Invalid dataset name'


def check_model_name(model_name):
    assert model_name in ['sim', 'pha', 'att'], 'Invalid model name'


def check_set_type(set_type, is_fold=False):
    if not is_fold:
        assert set_type in ['train', 'valid', 'evalu'], 'Invalid set type'
    else:
        assert set_type in ['train', 'evalu'], 'Invalid set type'
