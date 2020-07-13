import os
import numpy as np
import pandas as pd
from global_settings import DATA_FOLDER
w, s = 30, 20


def cat_generator(dataset, batch_size):
    assert dataset in ['ASAS', 'MACHO']
    catalog = pd.read_csv(os.path.join(DATA_FOLDER, 'ASAS', 'catalog.csv'), index_col=0)
    list(zip(catalog['Class'], catalog['Path']))
    for cats, paths in :
        X, y = [], []
        for cat, path in list(zip(cats, paths))
            df = pd.read_csv(os.path.join(DATA_FOLDER, 'ASAS', paths))
            X.append(processing(df)); y.append([cat])
        yield X, y


def cat_loader(dataset):
    assert dataset in ['ASAS', 'MACHO']
    catalog = pd.read_csv(os.path.join(DATA_FOLDER, 'ASAS', 'catalog.csv'), index_col=0)

    for cat, path in list(zip(catalog['Class'], catalog['Path'])):
        df = pd.read_csv(os.path.join(DATA_FOLDER, 'ASAS', path))


    return np.array(X), np.array(y)


def processing(df):
    df.sort_values(by=['mjd'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    mjd, mag = np.diff(df['mjd'].values).reshape(-1, 1), np.diff(df['mag'].values).reshape(-1, 1)
    dtdm_org = np.concatenate([mjd, mag], axis=1)
    dtdm_bin = np.array([], dtype=np.int64).reshape(0, 2 * w)
    for i in np.arange(np.shape(dtdm_org)[0] - w + 1)[::s]:
        dtdm_bin = np.vstack([dtdm_bin, dtdm_org[i: i + w, :].reshape(-1)])

    return dtdm_bin
