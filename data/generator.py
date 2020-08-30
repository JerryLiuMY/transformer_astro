import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tools.data_tools import load_sliding
from global_settings import RAW_FOLDER
from config.data_config import data_config
from data.core import load_dtdm, proc_dtdm
from sklearn.utils import class_weight
from data.loader import encoder_loader
window, stride = data_config['window'], data_config['stride']
ws, batch = data_config['ws'], data_config['batch']


def generator(dataset_name):
    dataset_name = dataset_name.decode("utf-8")
    (w, s) = ws[dataset_name]
    encoder = encoder_loader(dataset_name)
    sliding = load_sliding(dataset_name, 'train')

    for index, row in sliding.iterrows():
        pth, cat = row['Path'], row['Class']
        sta, end = row['Start'], row['End']
        data_df = pd.read_pickle(os.path.join(RAW_FOLDER, dataset_name, pth))

        dtdm_org = load_dtdm(data_df, sta, end)
        dtdm_bin = proc_dtdm(dtdm_org, w, s)
        x = np.expand_dims(dtdm_bin, axis=0)
        y = encoder.transform(np.expand_dims([cat], axis=0)).toarray()
        x, y = x.astype(np.float32), y.astype(np.float32)
        sample_weight = np.float32(class_weight.compute_sample_weight('balanced', y))

        yield x, y, sample_weight


def data_generator(dataset_name):
    (w, s) = ws[dataset_name]; steps = (window[dataset_name] - w) // s + 1
    encoder = encoder_loader(dataset_name)
    sliding = load_sliding(dataset_name, 'train')

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([None, steps, 2 * w]),
                       tf.TensorShape([None, len(encoder.categories_[0])]),
                       tf.TensorShape([None])),
        args=[dataset_name]
    )

    dataset = dataset.shuffle(np.shape(sliding)[0], reshuffle_each_iteration=True).batch(256)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()

    return dataset


def fold_generator(dataset_name, fold):
    pass
