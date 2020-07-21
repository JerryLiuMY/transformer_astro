import os
import math
import numpy as np
import pandas as pd
import pickle
import sklearn
from global_settings import DATA_FOLDER
from sklearn.utils import class_weight
from config.data_config import data_config
from config.train_config import train_config
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
window, stride, max_len = data_config['window'], data_config['stride'], data_config['max_len']
batch, sample = train_config['batch'], data_config['sample']


class DataGenerator(Sequence):
    def __init__(self, dataset_name):
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
        self.catalog = sklearn.utils.shuffle(self.catalog)

    def _data_generation(self, catalog_):
        x, y_spar = load_xy(self.dataset_name, catalog_)
        y = self.encoder.transform(y_spar).toarray()
        sample_weight = class_weight.compute_sample_weight('balanced', y)

        return x, y, sample_weight


def data_loader(dataset_name, set_type):
    assert set_type in ['whole', 'train', 'valid', 'evalu'], 'invalid set type'
    catalog = load_catalog(dataset_name, set_type)
    encoder = load_one_hot(dataset_name)
    print(f'{datetime.now()} Loading {dataset_name} {set_type} set')
    x, y_spar = load_xy(dataset_name, catalog)
    y = encoder.transform(y_spar).toarray()

    return x, y


def fold_generator(dataset_name):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    catalog = load_catalog(dataset_name, 'whole')
    encoder = load_one_hot(dataset_name)
    x, y_spar = load_xy(dataset_name, catalog)
    for train, test in skf.split(x, y_spar):
        pass


def load_catalog(dataset_name, set_type):
    assert dataset_name.split('_')[0] in ['ASAS', 'MACHO', 'WISE', 'GAIA', 'OGLE', 'Synthesis'], 'invalid dataset name'
    catalog = pd.read_csv(os.path.join(DATA_FOLDER, dataset_name, 'catalog.csv'), index_col=0)
    whole_catalog, train_catalog = pd.DataFrame(), pd.DataFrame()
    valid_catalog, evalu_catalog = pd.DataFrame(), pd.DataFrame()

    for cat in sorted(set(catalog['Class'])):
        catalog_ = catalog[catalog['Class'] == cat].reset_index(drop=True, inplace=False)
        np.random.seed(1); catalog_ = sklearn.utils.shuffle(catalog_); size_ = np.shape(catalog_)[0]
        whole_catalog = pd.concat([whole_catalog, catalog_])
        train_catalog = pd.concat([train_catalog, catalog_.iloc[:int(size_ * 0.7), :]])
        valid_catalog = pd.concat([valid_catalog, catalog_.iloc[int(size_ * 0.7): int(size_ * 0.8), :]])
        evalu_catalog = pd.concat([evalu_catalog, catalog_.iloc[int(size_ * 0.8):, :]])

    # train_catalog = pd.DataFrame()
    # for cat in sorted(set(unbal_catalog['Class'])):
    #     np.random.seed(1)
    #     train_catalog_ = unbal_catalog[unbal_catalog['Class'] == cat].reset_index(drop=True, inplace=False)
    #     train_catalog_ = train_catalog_.loc[choice(train_catalog_.index, sample, replace=True)]  # TODO: Change to False
    #     train_catalog = pd.concat([train_catalog, train_catalog_])

    catalog_dict = {'whole': whole_catalog.reset_index(drop=True),
                    'train': train_catalog.reset_index(drop=True),
                    'valid': valid_catalog.reset_index(drop=True),
                    'evalu': evalu_catalog.reset_index(drop=True)}

    return catalog_dict[set_type]


def load_one_hot(dataset_name):
    with open(os.path.join(DATA_FOLDER, dataset_name, 'encoder.pkl'), 'rb') as handle:
        encoder = pickle.load(handle)

    return encoder


def load_xy(dataset_name, catalog):
    cats, paths = list(catalog['Class']), list(catalog['Path'])

    x, y_spar = [], []
    for cat, path in list(zip(cats, paths)):
        data_df = pd.read_csv(os.path.join(DATA_FOLDER, dataset_name, path))
        x.append(processing(data_df)); y_spar.append([cat])
    x = pad_sequences(x, value=0.0, dtype=np.float64, maxlen=max_len, truncating='post', padding='post')

    return np.array(x), np.array(y_spar)


def processing(data_df):
    data_df.sort_values(by=['mjd'], inplace=True)
    data_df.reset_index(drop=True, inplace=True)
    mjd, mag = np.diff(data_df['mjd'].values).reshape(-1, 1), np.diff(data_df['mag'].values).reshape(-1, 1)
    dtdm_org = np.concatenate([mjd, mag], axis=1)
    dtdm_bin = np.array([], dtype=np.int64).reshape(0, 2 * window)
    for i in range(0, np.shape(dtdm_org)[0] - (window - 1), stride):
        dtdm_bin = np.vstack([dtdm_bin, dtdm_org[i: i + window, :].reshape(-1)])

    return dtdm_bin


def _dump_one_hot(dataset_name):
    catalog = load_catalog(dataset_name, 'whole')
    _, y_spar = load_xy(dataset_name, catalog)
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(y_spar)

    with open(os.path.join(DATA_FOLDER, dataset_name, 'encoder.pkl'), 'wb') as handle:
        pickle.dump(encoder, handle)


if __name__ == '__main__':
    pass
