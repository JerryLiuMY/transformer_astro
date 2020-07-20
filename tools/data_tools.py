import os
import numpy as np
import pandas as pd
import pickle
from global_settings import DATA_FOLDER
from config.data_config import data_config
from config.train_config import train_config
from sklearn.utils import shuffle
from numpy.random import choice
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

window, stride, max_len = data_config['window'], data_config['stride'], data_config['max_len']
batch, sample = train_config['batch'], data_config['sample']


def data_generator(dataset_name, set_type='train'):
    assert dataset_name.split('_')[0] in ['ASAS', 'MACHO', 'WISE', 'GAIA', 'OGLE', 'Synthesis'], 'invalid dataset name'
    catalog = load_catalog(dataset_name, set_type)
    encoder = load_one_hot(dataset_name)
    for i in range(0, np.shape(catalog)[0], batch):
        catalog_ = catalog.iloc[i: i+batch, :]
        x, y_spar = load_xy(dataset_name, catalog_, set_type)
        y = encoder.transform(y_spar).toarray()

        yield x, y


def data_loader(dataset_name, set_type):
    assert dataset_name.split('_')[0] in ['ASAS', 'MACHO', 'WISE', 'GAIA', 'OGLE', 'Synthesis'], 'invalid dataset name'
    assert set_type in ['whole', 'train', 'valid', 'evalu'], 'invalid set type'
    catalog = load_catalog(dataset_name, set_type)
    encoder = load_one_hot(dataset_name)
    x, y_spar = load_xy(dataset_name, catalog, set_type)
    y = encoder.transform(y_spar).toarray()

    return x, y_spar


def load_catalog(dataset_name, set_type):
    catalog = pd.read_csv(os.path.join(DATA_FOLDER, dataset_name, 'catalog.csv'), index_col=0)
    whole_catalog, unbal_catalog = pd.DataFrame(), pd.DataFrame()
    valid_catalog, evalu_catalog = pd.DataFrame(), pd.DataFrame()

    for cat in sorted(set(catalog['Class'])):
        catalog_ = catalog[catalog['Class'] == cat].reset_index(drop=True, inplace=False)
        np.random.seed(1); catalog_ = shuffle(catalog_); size_ = np.shape(catalog_)[0]
        whole_catalog = pd.concat([whole_catalog, catalog_])
        unbal_catalog = pd.concat([unbal_catalog, catalog_.iloc[:int(size_ * 0.7), :]])
        valid_catalog = pd.concat([valid_catalog, catalog_.iloc[int(size_ * 0.7): int(size_ * 0.8), :]])
        evalu_catalog = pd.concat([evalu_catalog, catalog_.iloc[int(size_ * 0.8):, :]])

    train_catalog = pd.DataFrame()
    for cat in sorted(set(unbal_catalog['Class'])):
        np.random.seed(1)
        train_catalog_ = unbal_catalog[unbal_catalog['Class'] == cat].reset_index(drop=True, inplace=False)
        # assert sample <= np.shape(train_catalog_)[0], 'Invalid sample size'
        train_catalog_ = train_catalog_.loc[choice(train_catalog_.index, sample, replace=True)]  # TODO: Change to False
        train_catalog = pd.concat([train_catalog, train_catalog_])

    catalog_dict = {'whole': whole_catalog.reset_index(drop=True),
                    'train': unbal_catalog.reset_index(drop=True),
                    'valid': valid_catalog.reset_index(drop=True),
                    'evalu': evalu_catalog.reset_index(drop=True)}

    return catalog_dict[set_type]


def load_one_hot(dataset_name):
    with open(os.path.join(DATA_FOLDER, dataset_name, 'encoder.pkl'), 'rb') as handle:
        encoder = pickle.load(handle)

    return encoder


def load_xy(dataset_name, catalog, set_type):
    print(f'{datetime.now()} Loading {dataset_name} {set_type} set')
    cats, paths = list(catalog['Class']), list(catalog['Path'])

    x, y_spar = [], []
    for cat, path in list(zip(cats, paths)):
        data_df = pd.read_csv(os.path.join(DATA_FOLDER, dataset_name, path))
        x.append(processing(data_df)); y_spar.append([cat])
    x = pad_sequences(x, value=0.0, dtype=np.float32, maxlen=max_len, truncating='post', padding='post')

    return x, y_spar


def processing(data_df):
    data_df.sort_values(by=['mjd'], inplace=True)
    data_df.reset_index(drop=True, inplace=True)
    mjd, mag = np.diff(data_df['mjd'].values).reshape(-1, 1), np.diff(data_df['mag'].values).reshape(-1, 1)
    dtdm_org = np.concatenate([mjd, mag], axis=1)
    dtdm_bin = np.array([], dtype=np.int64).reshape(0, 2 * window)
    for i in range(0, np.shape(dtdm_org)[0] - (window - 1), stride):
        dtdm_bin = np.vstack([dtdm_bin, dtdm_org[i: i + window, :].reshape(-1)])

    return dtdm_bin


def one_hot(dataset_name):
    catalog = load_catalog(dataset_name, 'whole')
    _, y_spar = load_xy(dataset_name, catalog, 'whole')
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(y_spar)

    with open(os.path.join(DATA_FOLDER, dataset_name, 'encoder.pkl'), 'wb') as handle:
        pickle.dump(encoder, handle)


if __name__ == '__main__':
    pass
