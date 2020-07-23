import math
import numpy as np
import sklearn
from sklearn.utils import class_weight
from config.data_config import data_config
from config.train_config import train_config
from tensorflow.keras.utils import Sequence
from datetime import datetime
from tools.utils import load_fold
from tools.utils import load_catalog, load_one_hot, load_xy

window, stride, max_len = data_config['window'], data_config['stride'], data_config['max_len']
batch, kfold, sample = train_config['batch'], train_config['kfold'], data_config['sample']


class DataGenerator(Sequence):
    def __init__(self, dataset_name):
        _check_dataset_name(dataset_name)
        self.dataset_name = dataset_name
        self.catalog = load_catalog(self.dataset_name, 'train')
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
        x, y_spar = load_xy(self.dataset_name, catalog_)
        y = self.encoder.transform(y_spar).toarray()
        sample_weight = class_weight.compute_sample_weight('balanced', y)

        return x, y, sample_weight


def data_loader(dataset_name, set_type):
    catalog = load_catalog(dataset_name, set_type)
    encoder = load_one_hot(dataset_name)
    print(f'{datetime.now()} Loading {dataset_name} {set_type} set')
    x, y_spar = load_xy(dataset_name, catalog)
    y = encoder.transform(y_spar).toarray()

    return x, y


def fold_loader(dataset_name, set_type, fold):
    fold_dict = load_fold(dataset_name, fold)
    train_catalog, valid_catalog = fold_dict['train'], fold_dict['valid']
    encoder = load_one_hot(dataset_name)
    print(f'{datetime.now()} Loading {dataset_name} fold {fold}')
    x_train, y_train_spar = load_xy(dataset_name, train_catalog)
    x_valid, y_valid_spar = load_xy(dataset_name, valid_catalog)
    y_train = encoder.transform(y_train_spar).toarray()
    y_valid = encoder.transform(y_valid_spar).toarray()

    return (x_train, y_train), (x_valid, y_valid)


def _check_dataset_name(dataset_name):
    assert dataset_name.split('_')[0] in ['ASAS', 'MACHO', 'WISE', 'GAIA', 'OGLE', 'Synthesis'], 'Invalid dataset name'


def _check_set_type(set_type):
    assert set_type in ['whole', 'train', 'valid', 'evalu'], 'Invalid set type'


if __name__ == '__main__':
    pass
