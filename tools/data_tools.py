import os
import numpy as np
import pandas as pd
from global_settings import DATA_FOLDER
from config.data_config import data_config

w, s, max_len, batch = data_config['w'], data_config['s'], data_config['max_len'], data_config['batch']


def cat_generator(dataset):
    assert dataset.split('_')[0] in ['ASAS', 'MACHO', 'Synthesis'], 'invalid dataset name'
    catalog = pd.read_csv(os.path.join(DATA_FOLDER, dataset, 'catalog.csv'), index_col=0)
    for i in range(0, np.shape(catalog)[0], batch):
        paths, cats = list(catalog['Path'])[i:i + batch], list(catalog['Class'])[i:i + batch]
        x, y = load_xy(dataset, cats, paths)

        yield np.array(x), np.array(y)


def cat_loader(dataset):
    assert dataset.split('_')[0] in ['ASAS', 'MACHO', 'Synthesis'], 'invalid dataset name'
    catalog = pd.read_csv(os.path.join(DATA_FOLDER, {dataset}, 'catalog.csv'), index_col=0)
    paths, cats = list(catalog['Path']), list(catalog['Class'])
    x, y = load_xy(dataset, cats, paths)

    return np.array(x), np.array(y)


def load_xy(dataset, cats, paths):
    x, y = [], []
    for cat, path in list(zip(cats, paths)):
        df = pd.read_csv(os.path.join(DATA_FOLDER, dataset, path))
        x.append(processing(df)); y.append([cat])  # TODO: Filter DataFrames shorter than 80

    return x, y


def processing(df):
    df.sort_values(by=['mjd'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    mjd, mag = np.diff(df['mjd'].values).reshape(-1, 1), np.diff(df['mag'].values).reshape(-1, 1)
    dtdm_org = np.concatenate([mjd, mag], axis=1)
    dtdm_bin = np.array([], dtype=np.int64).reshape(0, 2 * w)
    for i in range(0, np.shape(dtdm_org)[0] - (w - 1), s):
        dtdm_bin = np.vstack([dtdm_bin, dtdm_org[i: i + w, :].reshape(-1)])
        if np.shape(dtdm_bin)[0] == max_len: break

    return dtdm_bin


if __name__ == '__main__':
    pass
