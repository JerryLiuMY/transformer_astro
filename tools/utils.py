import os
import pickle
import numpy as np
import pandas as pd
import functools
import sklearn
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from global_settings import DATA_FOLDER
from config.data_config import data_config
from config.exec_config import evalu_config
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tqdm import tqdm
import multiprocessing

thresh, sample = data_config['thresh'], data_config['sample']
window, stride = data_config['window'], data_config['stride']
ws = data_config['ws']
kfold = evalu_config['kfold']


def load_catalog(dataset_name, set_type):
    catalog = pd.read_csv(os.path.join(DATA_FOLDER, dataset_name, 'catalog.csv'), index_col=0)
    whole_catalog, unbal_catalog = pd.DataFrame(), pd.DataFrame()
    valid_catalog, evalu_catalog = pd.DataFrame(), pd.DataFrame()
    encoder = pd.read_pickle(os.path.join(DATA_FOLDER, dataset_name, 'encoder.pkl'))
    cats = list(encoder.categories_[0])

    for cat in cats:
        catalog_ = catalog[catalog['Class'] == cat].reset_index(drop=True, inplace=False)
        catalog_ = sklearn.utils.shuffle(catalog_, random_state=0); size_ = np.shape(catalog_)[0]
        whole_catalog = pd.concat([whole_catalog, catalog_])
        unbal_catalog = pd.concat([unbal_catalog, catalog_.iloc[:int(size_ * 0.7), :]])
        valid_catalog = pd.concat([valid_catalog, catalog_.iloc[int(size_ * 0.7): int(size_ * 0.8), :]])
        evalu_catalog = pd.concat([evalu_catalog, catalog_.iloc[int(size_ * 0.8):, :]])

    ros = RandomOverSampler(sampling_strategy='auto', random_state=1)
    rus = RandomUnderSampler(sampling_strategy={cat: sample[dataset_name] for cat in cats}, random_state=1)
    unbal_catalog, _ = ros.fit_resample(unbal_catalog, unbal_catalog['Class'])
    train_catalog, _ = rus.fit_resample(unbal_catalog, unbal_catalog['Class'])

    catalog_dict = {'whole': whole_catalog.reset_index(drop=True),
                    'train': train_catalog.reset_index(drop=True),
                    'valid': valid_catalog.reset_index(drop=True),
                    'evalu': evalu_catalog.reset_index(drop=True)}

    return catalog_dict[set_type]


def load_fold(dataset_name, set_type, fold):
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=0)
    catalog = load_catalog(dataset_name, 'whole')
    y_spar = catalog['Class'].values.reshape(-1, 1)

    fold_dict = {}; fold_idx = 0
    for train_idx, valid_idx in skf.split(catalog, y_spar):
        # No need to reset index on each catalog
        train_catalog = catalog.iloc[train_idx, :]
        valid_catalog = catalog.iloc[valid_idx, :]
        fold_dict[str(fold_idx)] = {'train': train_catalog, 'valid': valid_catalog}
        fold_idx += 1

    return fold_dict[fold][set_type]


def load_xy(dataset_name, set_type, catalog):
    num_process = 8
    pool, catalogs = multiprocessing.Pool(num_process), np.array_split(catalog, num_process)

    x, y_spar, drop_count = [], [], 0
    for result in tqdm(pool.imap(functools.partial(load_xy_nest, dataset_name, set_type), catalogs)):
        x_, y_spar_, drop_count_ = result
        [x.append(foo) for foo in x_]
        [y_spar.append(bar) for bar in y_spar_]
        drop_count += drop_count_
    pool.close()
    pool.join()

    print(f'Number of dropped samples: {drop_count}')
    x = pad_sequences(x, value=np.pi, dtype=np.float32, padding='post')
    x, y_spar = np.array(x), np.array(y_spar)

    return x, y_spar


def load_xy_nest(dataset_name, set_type, catalog_):
    cats_, paths_ = list(catalog_['Class']), list(catalog_['Path'])
    x_, y_spar_, drop_count_ = [], [], 0
    for cat_, path_ in tqdm(list(zip(cats_, paths_))):
        data_df_ = pd.read_pickle(os.path.join(DATA_FOLDER, dataset_name, path_))

        if (set_type != 'evalu') and (np.shape(data_df_)[0] >= window[dataset_name]):
            x__, y_spar__ = _processing(dataset_name, cat_, data_df_)
            [x_.append(foo) for foo in x__]
            [y_spar_.append(bar) for bar in y_spar__]
        elif set_type == 'evalu':
            (w, s) = ws[dataset_name]
            x_.append(_proc_dtdm(_load_dtdm(data_df_), w, s))
            y_spar_.append([cat_])
        else:
            drop_count_ += 1

    return x_, y_spar_, drop_count_


def _processing(dataset_name, cat, data_df):
    dtdm_org, (w, s) = _load_dtdm(data_df), ws[dataset_name]
    x_, y_spar_ = np.array([]).reshape([0, (window[dataset_name]-w)//s + 1, 2 * w]), np.array([]).reshape([0, 1])
    for i in range(0, np.shape(dtdm_org)[0] - (window[dataset_name] - 1), stride[dataset_name]):
        dtdm_bin_ = _proc_dtdm(dtdm_org[i: i + window[dataset_name], :], w, s)
        dtdm_bin_, cat_ = np.expand_dims(dtdm_bin_, axis=0), np.expand_dims([cat], axis=0)
        x_, y_spar_ = np.vstack([x_, dtdm_bin_]), np.vstack([y_spar_, cat_])

    return x_, y_spar_


def _load_dtdm(data_df):
    scaler = MinMaxScaler(feature_range=(0, 30))
    scaler.fit(data_df['mag'].values.reshape(-1, 1))
    mjd = data_df['mjd'].values.reshape(-1, 1)
    mag = scaler.transform(data_df['mag'].values.reshape(-1, 1))
    mjd_diff, mag_diff = np.diff(mjd, axis=0), np.diff(mag, axis=0)
    dtdm_org = np.concatenate([mjd_diff, mag_diff], axis=1)

    return dtdm_org


def _proc_dtdm(dtdm_org, w, s):
    dtdm_bin = np.array([]).reshape(0, 2 * w)
    for i in range(0, np.shape(dtdm_org)[0] - (w - 1), s):
        dtdm_bin = np.vstack([dtdm_bin, dtdm_org[i: i + w, :].reshape(1, -1)])

    return dtdm_bin


def save_one_hot(dataset_name):
    catalog, cats = pd.read_csv(os.path.join(DATA_FOLDER, dataset_name, 'catalog.csv'), index_col=0), []
    for cat in sorted(set(catalog['Class'])):
        if len(catalog[catalog['Class'] == cat]) >= thresh[dataset_name]:
            cats.append(cat)
    catalog = catalog[catalog['Class'].isin(cats)].reset_index(drop=True, inplace=False)

    y_spar = np.array(sorted(list(catalog['Class']))).reshape(-1, 1)
    encoder = OneHotEncoder(handle_unknown='ignore', dtype=np.float32)
    encoder.fit(y_spar)

    with open(os.path.join(DATA_FOLDER, dataset_name, 'encoder.pkl'), 'wb') as handle:
        pickle.dump(encoder, handle)
