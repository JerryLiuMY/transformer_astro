import os
import math
import numpy as np
import sklearn
import pandas as pd
import pickle
from global_settings import DATA_FOLDER, DATASET_TYPE
from sklearn.utils import class_weight
from config.exec_config import train_config
from tensorflow.keras.utils import Sequence
from datetime import datetime
from tools.utils import load_catalog, load_fold
from tools.utils import load_one_hot, load_xy
from tools.utils import timer


batch = train_config['batch']


class BaseGenerator(Sequence):
    def __init__(self, dataset_name):
        _check_dataset_name(dataset_name)
        self.dataset_name = dataset_name
        self.catalog = pd.DataFrame()
        self.encoder = load_one_hot(self.dataset_name)
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
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.catalog = load_catalog(self.dataset_name, 'train')


class FoldGenerator(BaseGenerator):
    def __init__(self, dataset_name, fold):
        super().__init__(dataset_name)
        self.fold = fold
        self.catalog = load_fold(self.dataset_name, 'train', self.fold)


@timer
def data_loader(dataset_name, set_type):
    assert set_type in ['train', 'valid', 'evalu'], 'Invalid set type'
    dataset_folder = '_'.join([dataset_name, DATASET_TYPE]) if DATASET_TYPE is not None else dataset_name
    with open(os.path.join(DATA_FOLDER, dataset_folder, set_type + '.pkl'), 'rb') as handle:
        x, y = pickle.load(handle)

    return x, y


def fold_loader(dataset_name, set_type, fold):
    assert set_type in ['train', 'evalu'], 'Invalid set type'
    catalog = load_fold(dataset_name, set_type, fold)
    encoder = load_one_hot(dataset_name)
    print(f'{datetime.now()} Loading {dataset_name} {set_type} set fold {fold}')
    x, y_spar = load_xy(dataset_name, set_type, catalog)
    y = encoder.transform(y_spar).toarray()

    return x, y


def data_saver(dataset_name, set_type):
    assert set_type in ['train', 'valid', 'evalu'], 'Invalid set type'
    catalog = load_catalog(dataset_name, set_type)
    encoder = load_one_hot(dataset_name)
    print(f'{datetime.now()} Loading {dataset_name} {set_type} set')
    x, y_spar = load_xy(dataset_name, set_type, catalog)
    y = encoder.transform(y_spar).toarray()

    dataset_folder = '_'.join([dataset_name, DATASET_TYPE]) if DATASET_TYPE is not None else dataset_name
    with open(os.path.join(DATA_FOLDER, dataset_folder, set_type + '.pkl'), 'wb') as handle:
        pickle.dump((x, y), handle, protocol=4)


def _check_dataset_name(dataset_name):
    assert dataset_name.split('_')[0] in ['ASAS', 'MACHO', 'WISE', 'GAIA', 'OGLE', 'Synthesis'], 'Invalid dataset name'


if __name__ == '__main__':
    pass
