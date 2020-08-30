# import os
# import pickle
# import pandas as pd
# import numpy as np
# from tools.data_tools import load_catalog
# from datetime import datetime
# from tqdm import tqdm_notebook
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences
# from global_settings import RAW_FOLDER, DATA_FOLDER
# from data.core import _load_dtdm, _proc_dtdm
#
#
# def data_saver(dataset_name, set_type='analy'):
#     print(f'{datetime.now()} Loading {dataset_name} {set_type} set')
#     catalog = load_catalog(dataset_name, set_type)
#     encoder = pd.read_pickle(os.path.join(RAW_FOLDER, dataset_name, 'encoder.pkl'))
#     x, y_spar = load_xy(dataset_name, catalog)
#     y = encoder.transform(y_spar).toarray().astype(np.float32)
#     x, y = x.astype(np.float32), y.astype(np.float32)
#
#     with open(os.path.join(DATA_FOLDER, dataset_name, set_type + '.pkl'), 'wb') as handle:
#         pickle.dump((x, y), handle, protocol=4)
#
#
# def load_xy(dataset_name, catalog):
#     x, y_spar = [], []
#     for i in tqdm_notebook(range(len(catalog))):
#         pth, cat = list(catalog['Path'])[i], list(catalog['Class'])[i]
#         data_df = pd.read_pickle(os.path.join(RAW_FOLDER, dataset_name, pth))
#
#         dtdm_org = _load_dtdm(data_df)
#         dtdm_bin = _proc_dtdm(dataset_name, dtdm_org)
#         x.append(dtdm_bin)
#         y_spar.append(np.array([cat]))
#
#     x = pad_sequences(x, value=3.14159, dtype=np.float32, padding='post')
#     x, y_spar = np.array(x), np.array(y_spar)
#
#     return x, y_spar
