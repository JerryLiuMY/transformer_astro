import os
import itertools
import argparse
from tools.dir_tools import get_log_dir, get_exp_dir
from model._base import log_params
from model.lstm import SimpleLSTM, FoldSimpleLSTM
from model.transformer import Transformer, FoldTransformer
from config.exec_config import evalu_config
from tools.test_tools import get_exp
from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp
from test.evaluation import evaluate

kfold = evalu_config['kfold']


def run(dataset_name, model_name):
    print(f'Running experiments on {dataset_name} : {model_name}')
    log_dir = get_log_dir(dataset_name, model_name)
    exp_dir = get_exp_dir(log_dir); log_params(exp_dir)
    rnn_nums, rnn_dims, dnn_nums = rnn_nums_hp.domain.values, rnn_dims_hp.domain.values, dnn_nums_hp.domain.values
    for rnn_num, rnn_dim, dnn_num in itertools.product(rnn_nums, rnn_dims, dnn_nums):
        hyper_param = {rnn_nums_hp: rnn_num, rnn_dims_hp: rnn_dim, dnn_nums_hp: dnn_num}
        model = {'sim': SimpleLSTM, 'tra': Transformer}[model_name]
        exp = model(dataset_name, hyper_param, exp_dir=exp_dir)
        exp.run()


def test(dataset_name, model_name, exp_num):
    print(f'Running experiments on {dataset_name} : {model_name}')
    log_dir = get_log_dir(dataset_name, model_name)
    exp_dir = os.path.join(log_dir, f'experiment_{exp_num}')
    rnn_nums, rnn_dims, dnn_nums = rnn_nums_hp.domain.values, rnn_dims_hp.domain.values, dnn_nums_hp.domain.values
    for rnn_num, rnn_dim, dnn_num in itertools.product(rnn_nums, rnn_dims, dnn_nums):
        hyper_param = {rnn_nums_hp: rnn_num, rnn_dims_hp: rnn_dim, dnn_nums_hp: dnn_num}
        exp = get_exp(dataset_name, hyper_param, exp_dir=exp_dir, best_last='last')
        evaluate(exp)


def run_fold(dataset_name, model_name, hyper_param):
    log_dir = get_log_dir(dataset_name, model_name)
    fold_dir = os.path.join(log_dir, 'fold'); os.makedirs(fold_dir, exist_ok=False)
    for fold in map(str, range(0, 10)):
        fold_model = {'sim': FoldSimpleLSTM, 'tra': FoldTransformer}[model_name]
        fold_exp = fold_model(dataset_name, hyper_param, exp_dir=fold_dir, fold=fold)
        fold_exp.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('-d', '--dataset_name', type=str, help='dataset name')
    parser.add_argument('-m', '--model_name', type=str, help='dataset type')
    args = parser.parse_args()
    run(dataset_name=args.dataset_name, model_name=args.model_name)
