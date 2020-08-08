import os
import itertools
from global_settings import LOG_FOLDER
from tools.misc import new_dir
from model.base import log_params
from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp
from model.lstm import SimpleLSTM, FoldSimpleLSTM
from config.exec_config import evalu_config
import argparse

kfold = evalu_config['kfold']


def run(dataset_name, model_name):
    assert model_name in ['sim', 'pha', 'att']
    log_dir = os.path.join(LOG_FOLDER, f'{dataset_name}'); os.makedirs(log_dir, exist_ok=True)
    exp_dir = new_dir(log_dir); log_params(exp_dir)
    rnn_nums, rnn_dims, dnn_nums = rnn_nums_hp.domain.values, rnn_dims_hp.domain.values, dnn_nums_hp.domain.values
    for rnn_num, rnn_dim, dnn_num in itertools.product(rnn_nums, rnn_dims, dnn_nums):
        hyper_param = {rnn_nums_hp: rnn_num, rnn_dims_hp: rnn_dim, dnn_nums_hp: dnn_num}
        model = {'sim': SimpleLSTM}[model_name]  # 'pha': PhasedLSTM, 'att': Attention
        exp = model(dataset_name=dataset_name, hyper_param=hyper_param, exp_dir=exp_dir)
        exp.run()


def run_fold(dataset_name, model_name, hyper_param):
    for fold in map(str, range(0, 10)):
        exp_dir = os.path.join(LOG_FOLDER, f'{dataset_name}', f'fold_{fold}'); os.makedirs(exp_dir, exist_ok=False)
        fold_model = {'sim': FoldSimpleLSTM}[model_name]  # 'pha': FoldPhasedLSTM, 'att': FoldAttention
        fold_exp = fold_model(dataset_name=dataset_name, hyper_param=hyper_param, exp_dir=exp_dir, fold=fold)
        fold_exp.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('-d', '--dataset_name', type=str, help='dataset name')
    parser.add_argument('-m', '--model_name', type=str, help='dataset type')
    args = parser.parse_args()
    run(dataset_name=args.dataset_name, model_name=args.model_name)
