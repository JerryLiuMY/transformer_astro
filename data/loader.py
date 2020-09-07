import os
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime
from global_settings import DATA_FOLDER
from sklearn.utils import class_weight
from config.data_config import data_config
from data.core import load_xy
from tools.data_tools import load_fold
from tools.utils import encoder_msg, token_msg, data_msg
batch = data_config['batch']


@encoder_msg
def one_hot_loader(dataset_name):
    with open(os.path.join(DATA_FOLDER, dataset_name, 'encoder.pkl'), 'rb') as handle:
        one_hot = pickle.load(handle)

    return one_hot


@token_msg
def token_loader(dataset_name):
    with open(os.path.join(DATA_FOLDER, dataset_name, 'token.pkl'), 'rb') as handle:
        token = pickle.load(handle)

    return token


@data_msg
def data_loader(dataset_name, set_type):
    with open(os.path.join(DATA_FOLDER, dataset_name, set_type + '.pkl'), 'rb') as handle:
        x, y = pickle.load(handle)

    if set_type == 'train':
        sample_weight = np.float32(class_weight.compute_sample_weight('balanced', y))
        dataset = tf.data.Dataset.from_tensor_slices((x, y, sample_weight))
        dataset = dataset.shuffle(np.shape(x)[0], reshuffle_each_iteration=True).batch(batch)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.batch(batch)

    return dataset


def fold_loader(dataset_name, set_type, fold):
    print(f'{datetime.now()} Loading {dataset_name} {set_type} set fold {fold}')

    sliding = load_fold(dataset_name, set_type, fold)
    one_hot = one_hot_loader(dataset_name)
    x, y_spar = load_xy(dataset_name, sliding)
    y = one_hot.transform(y_spar).toarray().astype(np.float32)
    x, y = x.astype(np.float32), y.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(batch)

    return dataset
