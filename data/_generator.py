import math
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.utils import class_weight
from data.loader import encoder_loader, batch
from tensorflow.python.keras.utils.data_utils import Sequence
from tools.data_tools import load_sliding, load_fold
from data.core import load_xy


class BaseGenerator(Sequence):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.sliding = pd.DataFrame()
        self.encoder = encoder_loader(self.dataset_name)
        self.on_epoch_end()

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
        y = self.encoder.transform(y_spar).toarray()
        x, y = x.astype(np.float32), y.astype(np.float32)
        sample_weight = np.float32(class_weight.compute_sample_weight('balanced', y))
        dataset = tf.data.Dataset.from_tensor_slices((x, y, sample_weight))

        return dataset


class DataGenerator(BaseGenerator):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.sliding = load_sliding(self.dataset_name, 'train')


class FoldGenerator(BaseGenerator):
    def __init__(self, dataset_name, fold):
        super().__init__(dataset_name)
        self.fold = fold
        self.sliding = load_fold(self.dataset_name, 'train', self.fold)
