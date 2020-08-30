import functools
import multiprocessing
import os
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler
from global_settings import RAW_FOLDER
from config.data_config import data_config

window, stride = data_config['window'], data_config['stride']
ws = data_config['ws']


def load_xy(dataset_name, sliding):
    num_process = 8
    (w, s) = ws[dataset_name]; steps = (window[dataset_name] - w) // s + 1
    x, y_spar = np.array([]).reshape([0, steps, 2 * w]), np.array([]).reshape([0, 1])

    load_xy_nest_ = functools.partial(load_xy_nest, dataset_name)
    pool, slidings = multiprocessing.Pool(num_process), np.array_split(sliding, num_process)
    for result_ in tqdm_notebook(pool.imap(load_xy_nest_, slidings), total=num_process):
        x, y_spar = np.vstack([x, result_[0]]), np.vstack([y_spar, result_[1]])
    pool.close()
    pool.join()

    return x, y_spar


def load_xy_nest(dataset_name, sliding_):
    (w, s) = ws[dataset_name]; steps = (window[dataset_name] - w) // s + 1
    x_, y_spar_ = np.array([]).reshape([0, steps, 2 * w]), np.array([]).reshape([0, 1])

    # faster than df.iterrows() but memory consuming
    for index in range(len(sliding_)):
        if index % 1000 == 0:
            print(f'{datetime.now()} Finished {index} / {len(sliding_)}')

        pth, cat = list(sliding_['Path'])[index], list(sliding_['Class'])[index]
        sta, end = list(sliding_['Start'])[index], list(sliding_['End'])[index]
        data_df = pd.read_pickle(os.path.join(RAW_FOLDER, dataset_name, pth))

        dtdm_org = load_dtdm(data_df, sta, end)
        dtdm_bin = proc_dtdm(dtdm_org, w, s)
        dtdm_bin, cat = np.expand_dims(dtdm_bin, axis=0), np.expand_dims([cat], axis=0)
        x_, y_spar_ = np.vstack([x_, dtdm_bin]), np.vstack([y_spar_, cat])

    return x_, y_spar_


def load_dtdm(data_df, sta, end):
    scaler = MinMaxScaler(feature_range=(0, 30))
    scaler.fit(data_df['mag'].values.reshape(-1, 1))
    mjd = data_df['mjd'].values.reshape(-1, 1)
    mag = scaler.transform(data_df['mag'].values.reshape(-1, 1))
    mjd_diff, mag_diff = np.diff(mjd, axis=0), np.diff(mag, axis=0)
    dtdm_org = np.concatenate([mjd_diff, mag_diff], axis=1)
    dtdm_org = dtdm_org[sta: end, :]

    return dtdm_org


def proc_dtdm(dtdm_org, w, s):
    dtdm_bin = np.array([]).reshape(0, 2 * w)
    for i in range(0, np.shape(dtdm_org)[0] - (w - 1), s):
        dtdm_bin = np.vstack([dtdm_bin, dtdm_org[i: i + w, :].reshape(1, -1)])

    return dtdm_bin

