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


def load_sliding(dataset_name, set_type):
    window_stride = f'{window[dataset_name]}_{stride[dataset_name]}'
    cats = list(pd.read_pickle(os.path.join(DATA_FOLDER, dataset_name, 'encoder.pkl')).categories_[0])
    catalog = pd.read_csv(os.path.join(DATA_FOLDER, dataset_name, 'catalog.csv'), index_col=0)
    sliding = pd.read_csv(os.path.join(DATA_FOLDER, dataset_name, f'{window_stride}.csv'), index_col=0)
    whole_catalog, unbal_catalog = pd.DataFrame(), pd.DataFrame()
    valid_catalog, evalu_catalog = pd.DataFrame(), pd.DataFrame()

    for cat in cats:
        catalog_raw_ = sklearn.utils.shuffle(catalog[catalog['Class'] == cat], random_state=0)
        len_ = np.shape(catalog_raw_)[0]
        whole_catalog = pd.concat([whole_catalog, catalog_raw_])
        unbal_catalog = pd.concat([unbal_catalog, catalog_raw_.iloc[:int(len_ * 0.7), :]])
        valid_catalog = pd.concat([valid_catalog, catalog_raw_.iloc[int(len_ * 0.7): int(len_ * 0.8), :]])
        evalu_catalog = pd.concat([evalu_catalog, catalog_raw_.iloc[int(len_ * 0.8):, :]])

    whole_sliding = sliding.loc[sliding['Path'].isin(whole_catalog['Path'])]
    unbal_sliding = sliding.loc[sliding['Path'].isin(unbal_catalog['Path'])]
    valid_sliding = sliding.loc[sliding['Path'].isin(valid_catalog['Path'])]
    evalu_sliding = sliding.loc[sliding['Path'].isin(evalu_catalog['Path'])]

    ros = RandomOverSampler(sampling_strategy='auto', random_state=1)
    rus = RandomUnderSampler(sampling_strategy={cat: sample[dataset_name] for cat in cats}, random_state=1)
    unbal_sliding, _ = ros.fit_resample(unbal_sliding, unbal_sliding['Class'])
    train_catalog, _ = rus.fit_resample(unbal_sliding, unbal_sliding['Class'])

    sliding_dict = {'whole': whole_sliding.reset_index(drop=True),  # ordered
                    'train': train_catalog.reset_index(drop=True),  # shuffled
                    'valid': valid_sliding.reset_index(drop=True),  # ordered
                    'evalu': evalu_sliding.reset_index(drop=True)}  # ordered

    return sliding_dict[set_type]


def load_fold(dataset_name, set_type, fold):
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=0)
    catalog = load_sliding(dataset_name, 'whole')
    y_spar = catalog['Class'].values.reshape(-1, 1)

    fold_dict = {}; fold_idx = 0
    for train_idx, valid_idx in skf.split(catalog, y_spar):
        # No need to reset index on each catalog
        train_catalog = catalog.iloc[train_idx, :]
        valid_catalog = catalog.iloc[valid_idx, :]
        fold_dict[str(fold_idx)] = {'train': train_catalog, 'valid': valid_catalog}
        fold_idx += 1

    return fold_dict[fold][set_type]


def load_xy(dataset_name, set_type, sliding):
    num_process = 8
    pool, catalogs = multiprocessing.Pool(num_process), np.array_split(sliding, num_process)

    x, y_spar, drop_count = [], [], 0
    for result in tqdm(pool.imap(functools.partial(load_xy_nest, dataset_name, set_type), catalogs)):
        x_, y_spar_, drop_count_ = result
        [x.append(foo) for foo in x_]
        [y_spar.append(bar) for bar in y_spar_]
        drop_count += drop_count_
    pool.close()
    pool.join()

    print(f'Number of dropped samples: {drop_count}')
    x = pad_sequences(x, value=3.14159, dtype=np.float32, padding='post')
    x, y_spar = np.array(x), np.array(y_spar)

    return x, y_spar


def load_xy_nest(dataset_name, set_type, sliding_):
    cats_, paths_ = list(sliding_['Class']), list(sliding_['Path'])
    x_, y_spar_, drop_count_ = [], [], 0
    for cat_, path_ in tqdm(list(zip(cats_, paths_))):
        data_df_ = pd.read_pickle(os.path.join(DATA_FOLDER, dataset_name, path_))

        if (set_type in ['train', 'valid']) and (np.shape(data_df_)[0] >= window[dataset_name]):
            dtdm_org = _load_dtdm(data_df_)
            dtdm_bin = _proc_dtdm(dataset_name, dtdm_org[start: end, :])
        elif set_type == 'evalu':
            dtdm_org = _load_dtdm(data_df_)
            dtdm_bin = _proc_dtdm(dataset_name, dtdm_org)
            x_.append(dtdm_bin)
            y_spar_.append([cat_])
        else:
            drop_count_ += 1

    return x_, y_spar_, drop_count_


def _load_dtdm(data_df):
    scaler = MinMaxScaler(feature_range=(0, 30))
    scaler.fit(data_df['mag'].values.reshape(-1, 1))
    mjd = data_df['mjd'].values.reshape(-1, 1)
    mag = scaler.transform(data_df['mag'].values.reshape(-1, 1))
    mjd_diff, mag_diff = np.diff(mjd, axis=0), np.diff(mag, axis=0)
    dtdm_org = np.concatenate([mjd_diff, mag_diff], axis=1)

    return dtdm_org


def _proc_dtdm(dataset_name, dtdm_org):
    (w, s) = ws[dataset_name]
    dtdm_bin = np.array([]).reshape(0, 2 * w)
    for i in range(0, np.shape(dtdm_org)[0] - (w - 1), s):
        dtdm_bin = np.vstack([dtdm_bin, dtdm_org[i: i + w, :].reshape(1, -1)])

    return dtdm_bin


def save_one_hot(dataset_name):
    sliding, cats = load_sliding(dataset_name, 'whole'), []
    for cat in sorted(set(sliding['Class'])):
        if len(sliding[sliding['Class'] == cat]) >= thresh[dataset_name]:
            cats.append(cat)
    sliding = sliding[sliding['Class'].isin(cats)].reset_index(drop=True, inplace=False)

    y_spar = np.array(sorted(list(sliding['Class']))).reshape(-1, 1)
    encoder = OneHotEncoder(handle_unknown='ignore', dtype=np.float32)
    encoder.fit(y_spar)

    with open(os.path.join(DATA_FOLDER, dataset_name, 'encoder.pkl'), 'wb') as handle:
        pickle.dump(encoder, handle)
