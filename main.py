import os
import itertools
from global_settings import DATA_FOLDER
from tools.utils import new_dir
from model.base import log_params
from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp
from model.gru import SimpleGRU, FoldBasic
from config.exec_config import evalu_config
kfold = evalu_config['kfold']


def run(dataset_name):
    exp_dir = new_dir(os.path.join(DATA_FOLDER, f'{dataset_name}_log'))
    log_params(exp_dir)
    rnn_nums, rnn_dims, dnn_nums = rnn_nums_hp.domain.values, rnn_dims_hp.domain.values, dnn_nums_hp.domain.values
    for rnn_num, rnn_dim, dnn_num in itertools.product(rnn_nums, rnn_dims, dnn_nums):
        hyper_param = {rnn_nums_hp: rnn_num, rnn_dims_hp: rnn_dim, dnn_nums_hp: dnn_num}
        exp = SimpleGRU(dataset_name=dataset_name, hyper_param=hyper_param, exp_dir=exp_dir)
        exp.build()
        exp.run()


def run_fold(dataset_name, hyper_param):
    for fold in map(str, range(0, 10)):
        exp_dir = os.path.join(DATA_FOLDER, f'{dataset_name}_fold', f'fold_{fold}'); os.mkdir(exp_dir)
        fold_exp = FoldBasic(dataset_name=dataset_name, hyper_param=hyper_param, exp_dir=exp_dir, fold=fold)
        fold_exp.build()
        fold_exp.run()
