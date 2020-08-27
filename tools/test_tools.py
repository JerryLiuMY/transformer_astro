import os
from datetime import datetime
from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp
from global_settings import LOG_FOLDER
from model.lstm import SimpleLSTM


def get_exp(dataset_name, model_name, exp_num, hyper_param, best_last):
    # input check
    print(f'{datetime.now()} Making prediction')
    assert best_last in ['best', 'last'], 'Invalid best_last type'
    exp_dir = os.path.join(LOG_FOLDER, f'{dataset_name}_{model_name}', f'experiment_{exp_num}')

    # get che_path
    che_dir, exp_name = os.path.join(exp_dir, 'checks'), None
    exp_names = [foo for foo in os.listdir(che_dir) if not foo.startswith('.')]
    for bar in exp_names:
        con1 = (bar.split('-')[0] == f'rnn_num_{hyper_param[rnn_nums_hp]}')
        con2 = (bar.split('-')[1] == f'rnn_dim_{hyper_param[rnn_dims_hp]}')
        con3 = (bar.split('-')[2] == f'dnn_num_{hyper_param[dnn_nums_hp]}')
        if con1 and con2 and con3:
            exp_name = bar
    if exp_name is None: raise AssertionError('Invalid hyper parameters')
    che_path = os.path.join(che_dir, exp_name)

    # get mod_path
    mod_names = [foo for foo in os.listdir(che_path) if not foo.startswith('.')]
    bl_step, bl_vacc = 0, 0.0
    for bar in mod_names:
        bar = '.'.join(bar.split('.')[:-1])
        step, vacc = int(bar.split('-')[0].split('_')[1]), float(bar.split('-')[1].split('_')[2])
        if best_last == 'last':
            bl_step, bl_vacc = (step, vacc) if step > bl_step else (bl_step, bl_vacc)
        else:
            bl_step, bl_vacc = (step, vacc) if vacc > bl_vacc else (bl_step, bl_vacc)
    mod_path = os.path.join(che_path, f'epoch_{bl_step}-val_acc_{bl_vacc}.hdf5')

    # get model & experiment
    exp = SimpleLSTM(dataset_name, model_name, hyper_param, exp_dir=exp_dir)
    exp.exp_name = exp_name
    exp.model.load_weights(mod_path)

    return exp

