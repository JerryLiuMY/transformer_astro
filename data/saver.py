import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from data.core import load_xy
from tools.data_tools import load_sliding
from global_settings import RAW_FOLDER, DATA_FOLDER


def encoder_saver(dataset_name):
    encoder = pd.read_pickle(os.path.join(RAW_FOLDER, dataset_name, 'encoder.pkl'))
    with open(os.path.join(DATA_FOLDER, dataset_name, 'encoder.pkl'), 'wb') as handle:
        pickle.dump(encoder, handle)


def token_saver(dataset_name):
    token = np.array(load_sliding(dataset_name, 'evalu')['Path'])
    with open(os.path.join(DATA_FOLDER, dataset_name, 'token.pkl'), 'wb') as handle:
        pickle.dump(token, handle)


def data_saver(dataset_name, set_type):
    print(f'{datetime.now()} Loading {dataset_name} {set_type} set')
    sliding = load_sliding(dataset_name, set_type)
    encoder = pd.read_pickle(os.path.join(RAW_FOLDER, dataset_name, 'encoder.pkl'))
    x, y_spar = load_xy(dataset_name, sliding)
    y = encoder.transform(y_spar).toarray()
    x, y = x.astype(np.float32), y.astype(np.float32)

    with open(os.path.join(DATA_FOLDER, dataset_name, set_type + '.pkl'), 'wb') as handle:
        pickle.dump((x, y), handle, protocol=4)
