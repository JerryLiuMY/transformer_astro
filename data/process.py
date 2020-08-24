from global_settings import DATA_FOLDER
from tqdm import tqdm_notebook
from config.data_config import data_config
import pandas as pd
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
    window, stride = data_config['window'][dataset_name], data_config['stride'][dataset_name]
    old_catalog = pd.read_csv(os.path.join(os.path.join(DATA_FOLDER, dataset_name + '_', 'catalog.csv')), index_col=0)
    new_catalog = pd.DataFrame(columns=['Path', 'Class', 'N'])
    new_catalog_ws = pd.DataFrame(columns=['Path', 'Class', 'N', 'Start', 'End'])

    for index, row in tqdm_notebook(list(old_catalog.iterrows())):
        Path, Class = row['Path'], row['Class']
        N = len(pd.read_pickle(os.path.join(DATA_FOLDER, dataset_name, Path)))
        new_catalog = new_catalog.append(
            {'Path': Path, 'Class': Class, 'N': N},
            ignore_index=True)

        # N >= window is automatically ensured
        for Start in range(0, N - (window - 1), stride):
            new_catalog_ws = new_catalog_ws.append(
                {'Path': Path, 'Class': Class, 'N': N, 'Start': Start, 'End': Start + stride},
                ignore_index=True)

    new_catalog.to_csv(os.path.join(DATA_FOLDER, dataset_name, 'catalog.csv'))
    new_catalog_ws.to_csv(os.path.join(DATA_FOLDER, dataset_name, f'catalog_{window}_{stride}.csv'))
