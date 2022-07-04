import math
import numpy as np
import sklearn
import tensorflow as tf
from sklearn.utils import class_weight
from data.loader import one_hot_loader
from config.data_config import data_config
from tensorflow.python.keras.utils.data_utils import Sequence
from tools.data_tools import load_sliding, load_fold
from data.core import load_xy
batch = data_config['batch']


class _BaseGenerator(Sequence):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.one_hot = one_hot_loader(self.dataset_name)
        self.sliding = None

    def __len__(self):
        return math.ceil(np.shape(self.sliding)[0] / batch)

    def __getitem__(self, index):
        sliding_ = self.sliding.iloc[index * batch: (index + 1) * batch, :]
        dataset = self._data_generation(sliding_)

        return dataset

    def on_epoch_end(self):
        self.sliding = sklearn.utils.shuffle(self.sliding, random_state=0)

    def _data_generation(self, sliding_):
        x, y_spar = load_xy(self.dataset_name, sliding_)
        y = self.one_hot.transform(y_spar).toarray()
        x, y = x.astype(np.float32), y.astype(np.float32)
        sample_weight = np.float32(class_weight.compute_sample_weight('balanced', y))
        dataset = tf.data.Dataset.from_tensor_slices((x, y, sample_weight)).batch(batch)

        return dataset


class DataGenerator(_BaseGenerator):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.sliding = load_sliding(self.dataset_name, 'train')


class FoldGenerator(_BaseGenerator):
    def __init__(self, dataset_name, fold):
        super().__init__(dataset_name)
        self.fold = fold
        self.sliding = load_fold(self.dataset_name, 'train', self.fold)
