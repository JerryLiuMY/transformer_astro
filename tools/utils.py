import os
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from global_settings import DATA_FOLDER
from config.data_config import data_config
from config.exec_config import evalu_config
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook

window, stride = data_config['window'], data_config['stride']
max_len, sample = data_config['max_len'], data_config['sample']
kfold = evalu_config['kfold']


def load_catalog(dataset_name, set_type):
    catalog = pd.read_csv(os.path.join(DATA_FOLDER, dataset_name, 'catalog.csv'), index_col=0)
    whole_catalog, unbal_catalog = pd.DataFrame(), pd.DataFrame()
    valid_catalog, evalu_catalog = pd.DataFrame(), pd.DataFrame()

    # TODO: WISE need to drop some categories
    cats = sorted(set(catalog['Class']))
    for cat in cats:
        catalog_ = catalog[catalog['Class'] == cat].reset_index(drop=True, inplace=False)
        catalog_ = sklearn.utils.shuffle(catalog_, random_state=0); size_ = np.shape(catalog_)[0]
        whole_catalog = pd.concat([whole_catalog, catalog_])
        unbal_catalog = pd.concat([unbal_catalog, catalog_.iloc[:int(size_ * 0.7), :]])
        valid_catalog = pd.concat([valid_catalog, catalog_.iloc[int(size_ * 0.7): int(size_ * 0.8), :]])
        evalu_catalog = pd.concat([evalu_catalog, catalog_.iloc[int(size_ * 0.8):, :]])

    ros = RandomOverSampler(sampling_strategy='auto', random_state=1)
    rus = RandomUnderSampler(sampling_strategy={cat: sample for cat in cats}, random_state=1)
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


def load_one_hot(dataset_name):
    with open(os.path.join(DATA_FOLDER, dataset_name, 'encoder.pkl'), 'rb') as handle:
        encoder = pickle.load(handle)

    return encoder


def load_xy(dataset_name, catalog):
    cats, paths = list(catalog['Class']), list(catalog['Path'])

    x, y_spar = [], []
    for cat, path in list(zip(cats, paths)):
        data_df = pd.read_csv(os.path.join(DATA_FOLDER, dataset_name, path))
        x.append(processing(data_df)); y_spar.append([cat])
    x = pad_sequences(x, value=-10, dtype=np.float32, maxlen=max_len, truncating='post', padding='post')
    x, y_spar = np.array(x), np.array(y_spar)

    return x, y_spar


def processing(data_df):
    data_df.sort_values(by=['mjd'], inplace=True)
    data_df.reset_index(drop=True, inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 30))
    scaler.fit(data_df['mag'].values.reshape(-1, 1))
    mjd = data_df['mjd'].values.reshape(-1, 1)
    mag = scaler.transform(data_df['mag'].values.reshape(-1, 1))
    mjd_diff, mag_diff = np.diff(mjd, axis=0), np.diff(mag, axis=0)

    dtdm_org = np.concatenate([mjd_diff, mag_diff], axis=1)
    dtdm_bin = np.array([], dtype=np.float32).reshape(0, 2 * window)
    for i in range(0, np.shape(dtdm_org)[0] - (window - 1), stride):
        dtdm_bin = np.vstack([dtdm_bin, dtdm_org[i: i + window, :].reshape(-1)])

    return dtdm_bin


def new_dir(log_dir):
    past_dirs = next(os.walk(log_dir))[1]
    new_num = 0 if len(past_dirs) == 0 else np.max([int(past_dir.split('_')[-1]) for past_dir in past_dirs]) + 1
    exp_dir = os.path.join(log_dir, '_'.join(['experiment', str(new_num)]))
    os.mkdir(exp_dir)

    return exp_dir


def _dump_one_hot(dataset_name):
    catalog = load_catalog(dataset_name, 'whole')
    _, y_spar = load_xy(dataset_name, catalog)
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(y_spar)

    with open(os.path.join(DATA_FOLDER, dataset_name, 'encoder.pkl'), 'wb') as handle:
        pickle.dump(encoder, handle)


def _dump_scaler(dataset_name, feature_range=(0, 30)):
    catalog = load_catalog(dataset_name, 'whole')
    cats, paths = list(catalog['Class']), list(catalog['Path'])
    mag_full = np.array([])
    for cat, path in tqdm_notebook(list(zip(cats, paths))):
        data_df = pd.read_csv(os.path.join(DATA_FOLDER, dataset_name, path))
        mag_full = np.append(mag_full, np.array(data_df['mag']))
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(mag_full.reshape(-1, 1))

    with open(os.path.join(DATA_FOLDER, dataset_name, 'scaler.pkl'), 'wb') as handle:
        pickle.dump(scaler, handle)
