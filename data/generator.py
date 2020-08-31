import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tools.data_tools import load_sliding, load_fold
from global_settings import RAW_FOLDER
from config.data_config import data_config
from data.core import load_dtdm, proc_dtdm
from sklearn.utils import class_weight
from data.loader import encoder_loader
window, stride = data_config['window'], data_config['stride']
ws, batch = data_config['ws'], data_config['batch']


class _BaseGenerator:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.encoder = encoder_loader(self.dataset_name)
        self.sliding = load_sliding(self.dataset_name, 'train')

        self.window = window[self.dataset_name]
        self.stride = stride[self.dataset_name]
        (self.w, self.s) = ws[self.dataset_name]

    def get_dataset(self):
        steps = (self.window - self.w) // self.s + 1
        dataset = tf.data.Dataset.from_generator(
            self._data_generation,
            output_types=(tf.float32, tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([None, steps, 2 * self.w]),
                           tf.TensorShape([None, len(self.encoder.categories_[0])]),
                           tf.TensorShape([None]))
        )

        dataset = dataset.shuffle(np.shape(self.sliding)[0], reshuffle_each_iteration=True).batch(256)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()

        return dataset

    def _data_generation(self):
        for index, row in self.sliding.iterrows():
            print(index)
            pth, cat = row['Path'], row['Class']
            sta, end = row['Start'], row['End']
            data_df = pd.read_pickle(os.path.join(RAW_FOLDER, self.dataset_name, pth))

            dtdm_org = load_dtdm(data_df, sta, end)
            dtdm_bin = proc_dtdm(dtdm_org, self.w, self.s)
            x = np.expand_dims(dtdm_bin, axis=0)
            y = self.encoder.transform(np.expand_dims([cat], axis=0)).toarray()
            x, y = x.astype(np.float32), y.astype(np.float32)
            sample_weight = np.float32(class_weight.compute_sample_weight('balanced', y))

            yield x, y, sample_weight


class DataGenerator(_BaseGenerator):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.sliding = load_sliding(self.dataset_name, 'train')


class FoldGenerator(_BaseGenerator):
    def __init__(self, dataset_name, fold):
        super().__init__(dataset_name)
        self.fold = fold
        self.sliding = load_fold(self.dataset_name, 'train', self.fold)

