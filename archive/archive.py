import os
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from global_settings import DATA_FOLDER


def std(dataset_name='OGLE'):
    catalog = pd.read_csv(os.path.join(DATA_FOLDER, dataset_name, 'catalog.csv'))
    base = os.path.join(DATA_FOLDER, dataset_name, 'LCs')
    cats, paths, count = list(catalog['Class']), list(catalog['Path']), 0
    for cat, path in tqdm_notebook(list(zip(cats, paths))):
        content = pd.read_csv(os.path.join(base, path.split('/')[-1]), sep='\\s+')
        if np.shape(content)[1] > 3:
            count += 1


def duplicate(dataset_name='OGLE'):
    catalog = pd.read_csv(os.path.join(DATA_FOLDER, dataset_name, 'catalog.csv'))
    base = os.path.join(DATA_FOLDER, dataset_name, 'LCs')
    paths, dupli = list(catalog['Path']), 0
    for path in tqdm_notebook(paths):
        usecols, names = [0, 1, 2], ['mjd', 'mag', 'magerr']
        content = pd.read_csv(os.path.join(base, path.split('/')[-1]), usecols=usecols, names=names, sep='\\s+')
        if content.duplicated(subset=['mjd']).sum() != 0:
            dupli += 1
