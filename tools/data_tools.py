import os
import numpy as np
import pandas as pd
from global_settings import DATA_FOLDER
from config.data_config import data_config

w, s, batch_size = data_config['w'], data_config['s'], data_config['batch_size']


def cat_generator(dataset):
    assert dataset.split('_')[0] in ['ASAS', 'MACHO', 'Synthesis'], 'invalid dataset name'
    catalog = pd.read_csv(os.path.join(DATA_FOLDER, dataset, 'catalog.csv'), index_col=0)
    for i in range(0, np.shape(catalog)[0], batch_size):
        paths, cats = list(catalog['Path'])[i:i+batch_size], list(catalog['Class'])[i:i+batch_size]
        X, y = load_Xy(dataset, cats, paths)

        yield np.array(X), np.array(y)


def cat_loader(dataset):
    assert dataset.split('_')[0] in ['ASAS', 'MACHO', 'Synthesis'], 'invalid dataset name'
    catalog = pd.read_csv(os.path.join(DATA_FOLDER, {dataset}, 'catalog.csv'), index_col=0)
    paths, cats = list(catalog['Path']), list(catalog['Class'])
    X, y = load_Xy(dataset, cats, paths)

    return np.array(X), np.array(y)


def load_Xy(dataset, cats, paths):
    X, y = [], []
    for cat, path in list(zip(cats, paths)):
        df = pd.read_csv(os.path.join(DATA_FOLDER, dataset, path))
        X.append(processing(df)); y.append([cat])  # TODO: Filter DataFrames shorter than 80

    return X, y


def processing(df):
    df.sort_values(by=['mjd'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    mjd, mag = np.diff(df['mjd'].values).reshape(-1, 1), np.diff(df['mag'].values).reshape(-1, 1)
    dtdm_org = np.concatenate([mjd, mag], axis=1)
    dtdm_bin = np.array([], dtype=np.int64).reshape(0, 2 * w)
    for i in range(0, np.shape(dtdm_org)[0] - (w - 1), s):
        dtdm_bin = np.vstack([dtdm_bin, dtdm_org[i: i + w, :].reshape(-1)])

    return dtdm_bin


if __name__ == '__main__':
    pass
