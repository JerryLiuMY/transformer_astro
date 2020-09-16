import os
import numpy as np
import pandas as pd
import sklearn
import warnings
from sklearn.model_selection import StratifiedKFold
from global_settings import RAW_FOLDER
from config.data_config import data_config
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
warnings.simplefilter(action='ignore', category=FutureWarning)

thresh, sample = data_config['thresh'], data_config['sample']
window, stride = data_config['window'], data_config['stride']
ws = data_config['ws']
kfold = data_config['kfold']


def load_catalog(dataset_name, set_type):
    catalog = pd.read_csv(os.path.join(RAW_FOLDER, dataset_name, 'catalog.csv'), index_col=0)
    cats = list(pd.read_pickle(os.path.join(RAW_FOLDER, dataset_name, 'encoder.pkl')).categories_[0])
    whole_catalog, unbal_catalog = pd.DataFrame(), pd.DataFrame()
    valid_catalog, analy_catalog = pd.DataFrame(), pd.DataFrame()

    for cat in cats:
        catalog_raw_ = sklearn.utils.shuffle(catalog[catalog['Class'] == cat], random_state=0)
        len_ = np.shape(catalog_raw_)[0]
        whole_catalog = pd.concat([whole_catalog, catalog_raw_])
        unbal_catalog = pd.concat([unbal_catalog, catalog_raw_.iloc[:int(len_ * 0.7), :]])
        valid_catalog = pd.concat([valid_catalog, catalog_raw_.iloc[int(len_ * 0.7): int(len_ * 0.8), :]])
        analy_catalog = pd.concat([analy_catalog, catalog_raw_.iloc[int(len_ * 0.8):, :]])

    ros = RandomOverSampler(sampling_strategy='auto', random_state=1)
    rus = RandomUnderSampler(sampling_strategy={cat: sample[dataset_name] for cat in cats}, random_state=1)
    unbal_catalog, _ = ros.fit_resample(unbal_catalog, unbal_catalog['Class'])
    train_catalog, _ = rus.fit_resample(unbal_catalog, unbal_catalog['Class'])

    catalog_dict = {'whole': whole_catalog.reset_index(drop=True),  # ordered
                    'train': train_catalog.reset_index(drop=True),  # ordered
                    'valid': valid_catalog.reset_index(drop=True),  # ordered
                    'analy': analy_catalog.reset_index(drop=True)}  # ordered

    return catalog_dict[set_type]


def load_sliding(dataset_name, set_type):
    window_stride = f'{window[dataset_name]}_{stride[dataset_name]}'
    sliding = pd.read_csv(os.path.join(RAW_FOLDER, dataset_name, f'{window_stride}.csv'), index_col=0)
    whole_catalog, train_catalog = load_catalog(dataset_name, 'whole'), load_catalog(dataset_name, 'train')
    valid_catalog, evalu_catalog = load_catalog(dataset_name, 'valid'), load_catalog(dataset_name, 'analy')

    # sliding: paths excluding short sequence; catalog:
    whole_sliding = whole_catalog.merge(sliding, how='inner', on=['Path', 'Class', 'N'])
    train_sliding = train_catalog.merge(sliding, how='inner', on=['Path', 'Class', 'N'])
    valid_sliding = valid_catalog.merge(sliding, how='inner', on=['Path', 'Class', 'N'])
    evalu_sliding = evalu_catalog.merge(sliding, how='inner', on=['Path', 'Class', 'N'])

    sliding_dict = {'whole': whole_sliding.reset_index(drop=True),  # ordered
                    'train': train_sliding.reset_index(drop=True),  # shuffled
                    'valid': valid_sliding.reset_index(drop=True),  # ordered
                    'evalu': evalu_sliding.reset_index(drop=True)}  # ordered

    return sliding_dict[set_type]


def load_fold(dataset_name, set_type, fold):
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=0)
    sliding = load_sliding(dataset_name, 'whole')
    y_spar = sliding['Class'].values.reshape(-1, 1)

    fold_dict = {}; fold_idx = 0
    for train_idx, valid_idx in skf.split(sliding, y_spar):
        # No need to reset index on each catalog
        train_catalog = sliding.iloc[train_idx, :]
        valid_catalog = sliding.iloc[valid_idx, :]
        fold_dict[str(fold_idx)] = {'train': train_catalog, 'valid': valid_catalog}
        fold_idx += 1

    return fold_dict[fold][set_type]
