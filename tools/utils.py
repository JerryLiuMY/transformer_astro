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
from tqdm import tqdm

window, stride = data_config['window'], data_config['stride']
max_len, sample = data_config['max_len'], data_config['sample']
kfold = evalu_config['kfold']


def load_catalog(dataset_name, set_type):
    catalog = pd.read_csv(os.path.join(DATA_FOLDER, dataset_name, 'catalog.csv'), index_col=0)
    whole_catalog, train_catalog = pd.DataFrame(), pd.DataFrame()
    valid_catalog, evalu_catalog = pd.DataFrame(), pd.DataFrame()

    # TODO: WISE need to drop some categories

    for cat in sorted(set(catalog['Class'])):
        catalog_ = catalog[catalog['Class'] == cat].reset_index(drop=True, inplace=False)
        catalog_ = sklearn.utils.shuffle(catalog_, random_state=0); size_ = np.shape(catalog_)[0]
        whole_catalog = pd.concat([whole_catalog, catalog_])
        train_catalog = pd.concat([train_catalog, catalog_.iloc[:int(size_ * 0.7), :]])
        valid_catalog = pd.concat([valid_catalog, catalog_.iloc[int(size_ * 0.7): int(size_ * 0.8), :]])
        evalu_catalog = pd.concat([evalu_catalog, catalog_.iloc[int(size_ * 0.8):, :]])

    # This step is re-sampling
    # train_catalog = pd.DataFrame()
    # for cat in sorted(set(unbal_catalog['Class'])):
    #     np.random.seed(1)
    #     train_catalog_ = unbal_catalog[unbal_catalog['Class'] == cat].reset_index(drop=True, inplace=False)
    #     train_catalog_ = train_catalog_.loc[choice(train_catalog_.index, sample, replace=True)]# TODO: Change to False
    #     train_catalog = pd.concat([train_catalog, train_catalog_])

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
    x = pad_sequences(x, value=0.0, dtype=np.float64, maxlen=max_len, truncating='post', padding='post')
    x, y_spar = np.array(x), np.array(y_spar)

    return x, y_spar


def processing(data_df):
    data_df.sort_values(by=['mjd'], inplace=True)
    data_df.reset_index(drop=True, inplace=True)
    mjd, mag = np.diff(data_df['mjd'].values).reshape(-1, 1), np.diff(data_df['mag'].values).reshape(-1, 1)
    dtdm_org = np.concatenate([mjd, mag], axis=1)
    dtdm_bin = np.array([], dtype=np.int64).reshape(0, 2 * window)
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