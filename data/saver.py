import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from data.loader import one_hot_loader
from global_settings import DATA_FOLDER
from tools.data_tools import load_catalog, load_xy


def one_hot_saver(dataset_name, model_name):
    encoder = pd.read_pickle(os.path.join(DATA_FOLDER, dataset_name, 'encoder.pkl'))
    dataset_folder = '_'.join([dataset_name, model_name])
    with open(os.path.join(DATA_FOLDER, dataset_folder, 'encoder.pkl'), 'wb') as handle:
        pickle.dump(encoder, handle)


def data_saver(dataset_name, model_name, set_type):
    print(f'{datetime.now()} Loading {dataset_name} {set_type} set')

    catalog = load_catalog(dataset_name, set_type)
    encoder = one_hot_loader(dataset_name, model_name)
    x, y_spar = load_xy(dataset_name, set_type, catalog)
    y = encoder.transform(y_spar).toarray().astype(np.float32)
    x, y = x.astype(np.float32), y.astype(np.float32)

    dataset_folder = '_'.join([dataset_name, model_name])
    with open(os.path.join(DATA_FOLDER, dataset_folder, set_type + '.pkl'), 'wb') as handle:
        pickle.dump((x, y), handle, protocol=4)
