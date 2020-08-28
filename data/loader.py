import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from global_settings import DATA_FOLDER
from sklearn.utils import class_weight
from config.exec_config import train_config
from datetime import datetime
from tools.data_tools import load_sliding, load_fold
from tools.data_tools import load_xy
from tools.misc import one_hot_msg, data_msg

batch = train_config['batch']


@one_hot_msg
def one_hot_loader(dataset_name, model_name):
    dataset_folder = '_'.join([dataset_name, model_name])
    with open(os.path.join(DATA_FOLDER, dataset_folder, 'encoder.pkl'), 'rb') as handle:
        encoder = pickle.load(handle)

    return encoder


@data_msg
def data_loader(dataset_name, model_name, set_type):
    dataset_folder = '_'.join([dataset_name, model_name])
    with open(os.path.join(DATA_FOLDER, dataset_folder, set_type + '.pkl'), 'rb') as handle:
        x, y = pickle.load(handle)

    if set_type == 'train':
        sample_weight = class_weight.compute_sample_weight('balanced', y)
        dataset = tf.data.Dataset.from_tensor_slices((x, y, sample_weight))
        dataset = dataset.shuffle(np.shape(x)[0], reshuffle_each_iteration=True).batch(batch)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.batch(batch)

    return dataset


def one_hot_saver(dataset_name, model_name):
    encoder = pd.read_pickle(os.path.join(DATA_FOLDER, dataset_name, 'encoder.pkl'))
    dataset_folder = '_'.join([dataset_name, model_name])
    with open(os.path.join(DATA_FOLDER, dataset_folder, 'encoder.pkl'), 'wb') as handle:
        pickle.dump(encoder, handle)


def data_saver(dataset_name, model_name, set_type):
    print(f'{datetime.now()} Loading {dataset_name} {set_type} set')
    catalog = load_sliding(dataset_name, set_type)
    encoder = one_hot_loader(dataset_name, model_name)
    x, y_spar = load_xy(dataset_name, set_type, catalog)
    y = encoder.transform(y_spar).toarray().astype(np.float32)
    x, y = x.astype(np.float32), y.astype(np.float32)

    dataset_folder = '_'.join([dataset_name, model_name])
    with open(os.path.join(DATA_FOLDER, dataset_folder, set_type + '.pkl'), 'wb') as handle:
        pickle.dump((x, y), handle, protocol=4)


def fold_loader(dataset_name, model_name, set_type, fold):
    print(f'{datetime.now()} Loading {dataset_name} {set_type} set fold {fold}')

    catalog = load_fold(dataset_name, set_type, fold)
    encoder = one_hot_loader(dataset_name, model_name)
    x, y_spar = load_xy(dataset_name, set_type, catalog)
    y = encoder.transform(y_spar).toarray().astype(np.float32)
    x, y = x.astype(np.float32), y.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(batch)

    return dataset
