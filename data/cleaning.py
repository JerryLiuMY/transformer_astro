import os
import glob
import functools
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm_notebook
from datetime import datetime
from global_settings import DATA_FOLDER
from config.data_config import data_config

window, stride = data_config['window'], data_config['stride']


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
    catalog_old = pd.read_csv(os.path.join(os.path.join(DATA_FOLDER, dataset_name + '_', 'catalog.csv')), index_col=0)
    catalog = pd.DataFrame(columns=['Path', 'Class', 'N'])
    sliding = pd.DataFrame(columns=['Path', 'Class', 'N', 'Start', 'End'])

    pool, old_catalogs = multiprocessing.Pool(num_process), np.array_split(catalog_old, num_process)
    clean_catalog_nest_ = functools.partial(clean_catalog_nest, dataset_name)
    for results in tqdm_notebook(pool.imap(clean_catalog_nest_, old_catalogs), total=num_process):
        catalog_raw_, catalog_sub_ = results
        catalog = catalog.append(catalog_raw_, ignore_index=True)
        sliding = sliding.append(catalog_sub_, ignore_index=True)

    catalog.reset_index(drop=True, inplace=True)
    sliding.reset_index(drop=True, inplace=True)
    window_stride = f'{window[dataset_name]}_{stride[dataset_name]}'
    catalog.to_csv(os.path.join(DATA_FOLDER, dataset_name, 'catalog.csv'))
    sliding.to_csv(os.path.join(DATA_FOLDER, dataset_name, f'{window_stride}.csv'))


def clean_catalog_nest(dataset_name, catalog_old_):
    catalog_old_ = catalog_old_.reset_index(drop=True, inplace=False)
    catalog_ = pd.DataFrame(columns=['Path', 'Class', 'N'])
    sliding_ = pd.DataFrame(columns=['Path', 'Class', 'N', 'Start', 'End'])

    for index, row in catalog_old_.iterrows():
        if index != 0 and index % 1000 == 0:
            print(f'{datetime.now()} Finished {index} / {len(catalog_old_)}')

        Path, Class = row['Path'], row['Class']
        data_df = pd.read_pickle(os.path.join(DATA_FOLDER, dataset_name, Path)); N = len(data_df)
        catalog_ = catalog_.append({'Path': Path, 'Class': Class, 'N': N}, ignore_index=True)

        # N >= window is automatically ensured
        for Start in range(0, N - (window[dataset_name] - 1), stride[dataset_name]):
            sliding_ = sliding_.append(
                {'Path': Path, 'Class': Class, 'N': N, 'Start': Start, 'End': Start + stride[dataset_name]},
                ignore_index=True
            )

    return catalog_, sliding_
