import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from tqdm import tqdm_notebook
from global_settings import DATA_FOLDER
from sklearn.preprocessing import MinMaxScaler


def std(dataset_name='OGLE'):
    """Check the number of columns in OGLE dataset"""
    catalog = pd.read_csv(os.path.join(DATA_FOLDER, dataset_name, 'catalog.csv'))
    base = os.path.join(DATA_FOLDER, dataset_name, 'LCs')
    cats, paths, count = list(catalog['Class']), list(catalog['Path']), 0
    for cat, path in tqdm_notebook(list(zip(cats, paths))):
        content = pd.read_csv(os.path.join(base, path.split('/')[-1]), sep='\\s+')
        if np.shape(content)[1] > 3:
            count += 1


def duplicate(dataset_name='OGLE'):
    """Check mjd duplicate in dataset"""
    catalog = pd.read_csv(os.path.join(DATA_FOLDER, dataset_name, 'catalog.csv'))
    base = os.path.join(DATA_FOLDER, dataset_name, 'LCs')
    paths, dupli = list(catalog['Path']), 0
    for path in tqdm_notebook(paths):
        usecols, names = [0, 1, 2], ['mjd', 'mag', 'magerr']
        content = pd.read_csv(os.path.join(base, path.split('/')[-1]), usecols=usecols, names=names, sep='\\s+')
        if content.duplicated(subset=['mjd']).sum() != 0:
            dupli += 1


def scaler_saver(dataset_name, feature_range=(0, 30)):
    """Global min-max scalar """
    catalog = pd.read_csv(os.path.join(DATA_FOLDER, dataset_name, 'catalog.csv'), index_col=0)
    cats, paths = list(catalog['Class']), list(catalog['Path'])
    mag_full = np.array([])
    for cat, path in tqdm(list(zip(cats, paths))):
        data_df = pd.read_pickle(os.path.join(DATA_FOLDER, dataset_name, path))
        mag_full = np.append(mag_full, np.array(data_df['mag']))
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(mag_full.reshape(-1, 1))

    with open(os.path.join(DATA_FOLDER, dataset_name, 'scaler.pkl'), 'wb') as handle:
        pickle.dump(scaler, handle)
