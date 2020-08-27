from global_settings import DATA_FOLDER
from tqdm import tqdm_notebook
from config.data_config import data_config
from datetime import datetime
import multiprocessing
import pandas as pd
import numpy as np
import functools
import glob
import os


def clean_data(dataset_name):
    files = glob.glob(os.path.join(os.path.join(DATA_FOLDER, dataset_name + '_', 'LCs', '*.dat')))
    for file in tqdm_notebook(files):
        data_df = pd.read_pickle(file)
        data_df = data_df.drop_duplicates(subset=['mjd'], keep='first', inplace=False)
        data_df = data_df.sort_values(by=['mjd'], inplace=False)
        data_df = data_df.reset_index(drop=True, inplace=False)
        data_df.to_pickle(os.path.join(DATA_FOLDER, dataset_name, 'LCs', file.split('/')[-1]))


def clean_catalog(dataset_name):
    num_process = 8
    window, stride = data_config['window'][dataset_name], data_config['stride'][dataset_name]
    old_catalog = pd.read_csv(os.path.join(os.path.join(DATA_FOLDER, dataset_name + '_', 'catalog.csv')), index_col=0)
    new_catalog = pd.DataFrame(columns=['Path', 'Class', 'N'])
    new_catalog_ws = pd.DataFrame(columns=['Path', 'Class', 'N', 'Start', 'End'])

    pool, old_catalogs = multiprocessing.Pool(num_process), np.array_split(old_catalog, num_process)
    clean_catalog_nest_ = functools.partial(clean_catalog_nest, dataset_name, window, stride)
    for results in tqdm_notebook(pool.imap(clean_catalog_nest_, old_catalogs), total=num_process):
        new_catalog_, new_catalog_ws_ = results
        new_catalog = new_catalog.append(new_catalog_, ignore_index=True)
        new_catalog_ws = new_catalog_ws.append(new_catalog_ws_, ignore_index=True)

    new_catalog.reset_index(drop=True, inplace=True)
    new_catalog_ws.reset_index(drop=True, inplace=True)
    new_catalog.to_csv(os.path.join(DATA_FOLDER, dataset_name, 'catalog.csv'))
    new_catalog_ws.to_csv(os.path.join(DATA_FOLDER, dataset_name, f'catalog_{window}_{stride}.csv'))


def clean_catalog_nest(dataset_name, window, stride, old_catalog_):
    old_catalog_ = old_catalog_.reset_index(drop=True, inplace=False)
    new_catalog_ = pd.DataFrame(columns=['Path', 'Class', 'N'])
    new_catalog_ws_ = pd.DataFrame(columns=['Path', 'Class', 'N', 'Start', 'End'])

    for index, row in old_catalog_.iterrows():
        if index != 0 and index % 1000 == 0:
            print(f'{datetime.now()} Finished {index} / {len(old_catalog_)}')

        Path, Class = row['Path'], row['Class']
        data_df = pd.read_pickle(os.path.join(DATA_FOLDER, dataset_name, Path))
        N = len(data_df)
        new_catalog_ = new_catalog_.append(
            {'Path': Path, 'Class': Class, 'N': N},
            ignore_index=True)

        # N >= window is automatically ensured
        for Start in range(0, N - (window - 1), stride):
            new_catalog_ws_ = new_catalog_ws_.append(
                {'Path': Path, 'Class': Class, 'N': N, 'Start': Start, 'End': Start + stride},
                ignore_index=True)

    return new_catalog_, new_catalog_ws_
