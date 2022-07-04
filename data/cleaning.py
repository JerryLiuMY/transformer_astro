import os
import glob
import functools
import pickle
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm_notebook
from datetime import datetime
from global_settings import RAW_FOLDER
from config.data_config import data_config
from sklearn.preprocessing import OneHotEncoder
from tools.data_tools import load_sliding, thresh

window, stride = data_config['window'], data_config['stride']


def clean_data(dataset_name):
    """save cleaned data to the raw directory"""
    files = glob.glob(os.path.join(os.path.join(RAW_FOLDER, dataset_name + '_', 'LCs', '*.dat')))
    for file in tqdm_notebook(files):
        data_df = pd.read_pickle(file)
        data_df = data_df.drop_duplicates(subset=['mjd'], keep='first', inplace=False)
        data_df = data_df.sort_values(by=['mjd'], inplace=False)
        data_df = data_df.reset_index(drop=True, inplace=False)
        data_df.to_pickle(os.path.join(RAW_FOLDER, dataset_name, 'LCs', file.split('/')[-1]))


def clean_catalog(dataset_name):
    """save catalog to the raw directory"""
    num_process = 8
    catalog_old = pd.read_csv(os.path.join(os.path.join(RAW_FOLDER, dataset_name + '_', 'catalog.csv')), index_col=0)
    catalog = pd.DataFrame(columns=['Path', 'Class', 'N'])

    pool, old_catalogs = multiprocessing.Pool(num_process), np.array_split(catalog_old, num_process)
    clean_catalog_nest_ = functools.partial(clean_catalog_nest, dataset_name)
    for catalog_ in tqdm_notebook(pool.imap(clean_catalog_nest_, old_catalogs), total=num_process):
        catalog = catalog.append(catalog_, ignore_index=True)

    catalog.reset_index(drop=True, inplace=True)
    catalog.to_csv(os.path.join(RAW_FOLDER, dataset_name, 'catalog.csv'))


def clean_catalog_nest(dataset_name, catalog_old_):
    catalog_old_ = catalog_old_.reset_index(drop=True, inplace=False)
    catalog_ = pd.DataFrame(columns=['Path', 'Class', 'N'])

    for index, row in catalog_old_.iterrows():
        if index != 0 and index % 1000 == 0:
            print(f'{datetime.now()} Finished {index} / {len(catalog_old_)}')

        Path, Class = row['Path'], row['Class']
        data_df = pd.read_pickle(os.path.join(RAW_FOLDER, dataset_name, Path)); N = len(data_df)
        catalog_ = catalog_.append(
            {'Path': Path, 'Class': Class, 'N': N}, ignore_index=True
        )

    return catalog_


def build_sliding(dataset_name):
    """save sliding to the raw directory"""
    num_process = 8
    catalog = pd.read_csv(os.path.join(os.path.join(RAW_FOLDER, dataset_name, 'catalog.csv')), index_col=0)
    sliding = pd.DataFrame(columns=['Path', 'Class', 'N', 'Start', 'End'])

    pool, slidings = multiprocessing.Pool(num_process), np.array_split(catalog, num_process)
    build_sliding_nest_ = functools.partial(build_sliding_nest, dataset_name)
    for sliding_ in tqdm_notebook(pool.imap(build_sliding_nest_, slidings), total=num_process):
        sliding = sliding.append(sliding_, ignore_index=True)

    sliding.reset_index(drop=True, inplace=True)
    window_stride = f'{window[dataset_name]}_{stride[dataset_name]}'
    sliding.to_csv(os.path.join(RAW_FOLDER, dataset_name, f'{window_stride}.csv'))


def build_sliding_nest(dataset_name, catalog_):
    catalog_ = catalog_.reset_index(drop=True, inplace=False)
    sliding_ = pd.DataFrame(columns=['Path', 'Class', 'N', 'Start', 'End'])

    for index, row in catalog_.iterrows():
        if index % 1000 == 0:
            print(f'{datetime.now()} Finished {index} / {len(catalog_)}')

        Path, Class, N = row['Path'], row['Class'], row['N']
        # - 1: for time difference; automatically N >= window
        for Start in range(0, N - 1 - (window[dataset_name] - 1), stride[dataset_name]):
            sliding_ = sliding_.append(
                {'Path': Path, 'Class': Class, 'N': N, 'Start': Start, 'End': Start + window[dataset_name]},
                ignore_index=True
            )

    return sliding_


def save_encoder(dataset_name):
    """save encoder to the raw directory"""
    sliding, cats = load_sliding(dataset_name, 'whole'), []
    for cat in sorted(set(sliding['Class'])):
        if len(sliding[sliding['Class'] == cat]) >= thresh[dataset_name]:
            cats.append(cat)
    sliding = sliding[sliding['Class'].isin(cats)].reset_index(drop=True, inplace=False)

    y_spar = np.array(sorted(list(sliding['Class']))).reshape(-1, 1)
    encoder = OneHotEncoder(handle_unknown='ignore', dtype=np.float32)
    encoder.fit(y_spar)

    with open(os.path.join(RAW_FOLDER, dataset_name, 'encoder.pkl'), 'wb') as handle:
        pickle.dump(encoder, handle)
