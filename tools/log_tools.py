import os
import numpy as np
import tensorflow as tf
from global_settings import LOG_FOLDER
from tensorflow.python.keras.callbacks import ReduceLROnPlateau


def create_dirs(*args):
    for path in args:
        if not os.path.isdir(path):
            os.mkdir(path)


create_paths = create_dirs


def get_exp_dir(log_dir):
    past_dirs = next(os.walk(log_dir))[1]
    new_num = 0 if len(past_dirs) == 0 else np.max([int(past_dir.split('_')[-1]) for past_dir in past_dirs]) + 1
    exp_dir = os.path.join(log_dir, '_'.join(['experiment', str(new_num)]))
    os.mkdir(exp_dir)

    return exp_dir


def get_log_dir(dataset_name, model_name):
    log_dir = os.path.join(LOG_FOLDER, f'{dataset_name}_{model_name}')
    os.makedirs(log_dir, exist_ok=True)

    return log_dir


def lnr_schedule(step):
    begin_rate = 0.001
    decay_rate = 0.7
    decay_step = 50

    learn_rate = begin_rate * np.power(decay_rate, np.divmod(step, decay_step)[0])
    tf.summary.scalar('Learning Rate', data=learn_rate, step=step)

    return learn_rate


def rop_schedule():
    rop_callback = ReduceLROnPlateau(
        monitor='val_loss',
        min_delta=0.001,
        factor=0.5,
        mode='min',
        patience=10,
        cooldown=5,
        verbose=1,
    )

    return rop_callback

