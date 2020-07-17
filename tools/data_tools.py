import os
import numpy as np
import pandas as pd
from global_settings import DATA_FOLDER
from config.data_config import data_config
from numpy.random import choice
from config.train_config import train_config

w, s, max_len = data_config['w'], data_config['s'], data_config['max_len']
batch = train_config['batch']


def data_generator(dataset_name, set_type='train'):
    catalog = load_catalog(dataset_name, set_type)
    for i in range(0, np.shape(catalog)[0], batch):
        catalog_ = catalog.iloc[i: i+batch, :]
        x, y = load_xy(dataset_name, catalog_)

        yield np.array(x), np.array(y)


def data_loader(dataset_name, set_type):
    catalog = load_catalog(dataset_name, set_type)
    x, y = load_xy(dataset_name, catalog)

    return np.array(x), np.array(y)


def load_catalog(dataset_name, set_type):
    assert dataset_name.split('_')[0] in ['ASAS', 'MACHO', 'WISE', 'GAIA', 'Synthesis'], 'invalid dataset name'
    assert set_type in ['train', 'valid', 'evalu'], 'invalid set type'
    catalog = pd.read_csv(os.path.join(DATA_FOLDER, dataset_name, 'catalog.csv'), index_col=0)
    train_catalog, valid_catalog, evalu_catalog = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for cat in sorted(set(catalog['Class'])):
        catalog_ = catalog[catalog['Class'] == cat].reset_index(drop=True, inplace=False)
        np.random.seed(1); size_ = np.shape(catalog_)[0]
        train_catalog = pd.concat([train_catalog, catalog_.loc[choice(catalog_.index, int(size_ * 0.7))]])
        valid_catalog = pd.concat([valid_catalog, catalog_.loc[choice(catalog_.index, int(size_ * 0.2))]])
        evalu_catalog = pd.concat([evalu_catalog, catalog_.loc[choice(catalog_.index, int(size_ * 0.1))]])

    catalog_dict = {'train': train_catalog.reset_index(drop=True),
                    'valid': valid_catalog.reset_index(drop=True),
                    'evalu': evalu_catalog.reset_index(drop=True)}

    return catalog_dict[set_type]


def load_xy(dataset_name, catalog):
    x, y = [], []
    cats, paths = list(catalog['Class']), list(catalog['Path'])
    for cat, path in list(zip(cats, paths)):
        data_df = pd.read_csv(os.path.join(DATA_FOLDER, dataset_name, path))
        x.append(processing(data_df)); y.append([cat])

    return x, y


def processing(data_df):
    data_df.sort_values(by=['mjd'], inplace=True)
    data_df.reset_index(drop=True, inplace=True)
    mjd, mag = np.diff(data_df['mjd'].values).reshape(-1, 1), np.diff(data_df['mag'].values).reshape(-1, 1)
    dtdm_org = np.concatenate([mjd, mag], axis=1)
    dtdm_bin = np.array([], dtype=np.int64).reshape(0, 2 * w)
    for i in range(0, np.shape(dtdm_org)[0] - (w - 1), s):
        dtdm_bin = np.vstack([dtdm_bin, dtdm_org[i: i + w, :].reshape(-1)])
        if np.shape(dtdm_bin)[0] == max_len: break

    return dtdm_bin


if __name__ == '__main__':
    pass
