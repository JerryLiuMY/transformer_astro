import os
import math
import numpy as np
import sklearn
import pandas as pd
import pickle
from global_settings import DATA_FOLDER
from sklearn.utils import class_weight
from config.exec_config import train_config
from tensorflow.keras.utils import Sequence
from datetime import datetime
from tools.utils import load_catalog, load_fold
from tools.utils import load_xy
from tools.misc import timer, check_model_name, check_set_type
batch = train_config['batch']


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


@timer
def data_loader(dataset_name, model_name, set_type):
    check_model_name(model_name); check_set_type(set_type)
    dataset_folder = '_'.join([dataset_name, model_name])
    with open(os.path.join(DATA_FOLDER, dataset_folder, set_type + '.pkl'), 'rb') as handle:
        x, y = pickle.load(handle)

    return x, y


def fold_loader(dataset_name, model_name, set_type, fold):
    check_model_name(model_name); check_set_type(set_type, is_fold=True)
    print(f'{datetime.now()} Loading {dataset_name} {set_type} set fold {fold}')

    catalog = load_fold(dataset_name, set_type, fold)
    encoder = one_hot_loader(dataset_name, model_name)
    x, y_spar = load_xy(dataset_name, set_type, catalog)
    y = encoder.transform(y_spar).toarray()

    return x, y


def one_hot_loader(dataset_name, model_name):
    dataset_folder = '_'.join([dataset_name, model_name])
    with open(os.path.join(DATA_FOLDER, dataset_folder, 'encoder.pkl'), 'rb') as handle:
        encoder = pickle.load(handle)

    return encoder


def data_saver(dataset_name, model_name, set_type):
    check_model_name(model_name); check_set_type(set_type)
    print(f'{datetime.now()} Loading {dataset_name} {set_type} set')

    catalog = load_catalog(dataset_name, set_type)
    encoder = one_hot_loader(dataset_name, model_name)
    x, y_spar = load_xy(dataset_name, set_type, catalog)
    y = encoder.transform(y_spar).toarray()

    dataset_folder = '_'.join([dataset_name, model_name])
    with open(os.path.join(DATA_FOLDER, dataset_folder, set_type + '.pkl'), 'wb') as handle:
        pickle.dump((x, y), handle, protocol=4)


