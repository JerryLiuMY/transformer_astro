import os
import pickle
import pandas as pd
import numpy as np
from tools.data_tools import load_catalog
from datetime import datetime
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from global_settings import RAW_FOLDER, DATA_FOLDER
from data.core import load_dtdm, proc_dtdm
from config.data_config import data_config

window, stride = data_config['window'], data_config['stride']
ws = data_config['ws']


def data_saver(dataset_name, set_type='analy'):
    print(f'{datetime.now()} Loading {dataset_name} {set_type} set')
    catalog = load_catalog(dataset_name, set_type)
    encoder = pd.read_pickle(os.path.join(RAW_FOLDER, dataset_name, 'encoder.pkl'))
    x, y_spar = load_xy(dataset_name, catalog)
    y = encoder.transform(y_spar).toarray().astype(np.float32)
    x, y = x.astype(np.float32), y.astype(np.float32)

    with open(os.path.join(DATA_FOLDER, dataset_name, set_type + '.pkl'), 'wb') as handle:
        pickle.dump((x, y), handle, protocol=4)


def load_xy(dataset_name, catalog):
    x, y_spar = [], []
    for index, row in catalog.iterrows():
        pth, cat = row['Path'], row['Class']
        data_df = pd.read_pickle(os.path.join(RAW_FOLDER, dataset_name, pth))

        (w, s) = ws[dataset_name]
        dtdm_org = load_dtdm(data_df, 0, len(data_df))
        dtdm_bin = proc_dtdm(dtdm_org, w, s)
        x.append(dtdm_bin)
        y_spar.append(np.array([cat]))

    x = pad_sequences(x, value=3.14159, dtype=np.float32, padding='post')
    x, y_spar = np.array(x), np.array(y_spar)

    return x, y_spar
