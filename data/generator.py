import math

import numpy as np
import pandas as pd
import sklearn
from sklearn.utils import class_weight
from tensorflow.python.keras.utils.data_utils import Sequence

from data.loader import one_hot_loader, batch
from tools.data_tools import load_xy, load_catalog, load_fold


class BaseGenerator(Sequence):
    def __init__(self, dataset_name, model_name):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.catalog = pd.DataFrame()
        self.encoder = one_hot_loader(self.dataset_name, self.model_name)
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(np.shape(self.catalog)[0] / batch)

    def __getitem__(self, index):
        catalog_ = self.catalog.iloc[index * batch: (index + 1) * batch, :]
        x, y, sample_weight = self._data_generation(catalog_)

        return x, y, sample_weight

    def on_epoch_end(self):
        self.catalog = sklearn.utils.shuffle(self.catalog, random_state=0)

    def _data_generation(self, catalog_):
        x, y_spar = load_xy(self.dataset_name, 'train', catalog_)
        y = self.encoder.transform(y_spar).toarray()
        sample_weight = class_weight.compute_sample_weight('balanced', y)
        x, y = x.astype(np.float32), y.astype(np.float32)

        return x, y, sample_weight


class DataGenerator(BaseGenerator):
    def __init__(self, dataset_name, model_name):
        super().__init__(dataset_name, model_name)
        self.catalog = load_catalog(self.dataset_name, 'train')


class FoldGenerator(BaseGenerator):
    def __init__(self, dataset_name, model_name, fold):
        super().__init__(dataset_name, model_name)
        self.fold = fold
        self.catalog = load_fold(self.dataset_name, 'train', self.fold)