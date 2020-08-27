import os
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Model
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from model.lstm import SimpleLSTM
from global_settings import LOG_FOLDER
from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp
sns.set()


def plot_timestep(dataset_name, model_name, exp_num, hyper_param, best_last):
    # get model
    assert best_last in ['best', 'last'], 'Invalid best_last type'
    exp_dir = os.path.join(LOG_FOLDER, f'{dataset_name}_{model_name}', f'experiment_{exp_num}')
    mod_path = get_path(exp_dir, hyper_param, best_last)
    with tf.device('/CPU:0'):
        exp = SimpleLSTM(dataset_name, model_name, hyper_param, exp_dir=exp_dir)
        exp.model.load_weights(mod_path)
        model = Model(inputs=exp.model.inputs, outputs=exp.model.get_layer('softmax').output)

    # evaluate accuracy
    y_evalu = np.array([]).reshape(0, len(exp.categories))
    for x_evalu_, y_evalu_ in exp.dataset_evalu.take(-1):
        y_evalu = np.vstack([y_evalu, y_evalu_.numpy()])
    y_predi_seq = model.predict(exp.dataset_evalu)
    acc_seq, metric = np.array([]), tf.keras.metrics.CategoricalAccuracy()
    for step in tqdm_notebook(range(np.shape(y_predi_seq)[1])):
        metric.update_state(y_evalu, y_predi_seq[:, step, :])
        acc_seq = np.append(acc_seq, metric.result().numpy())

    # plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(acc_seq)
    ax.set_xlabel('time step')
    ax.set_ylabel('test categorical accuracy')

    return fig


def get_path(exp_dir, hyper_param, best_last):
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

    return mod_path


if __name__ == '__main__':
    plot_timestep('ASAS', 'sim', '10', {rnn_nums_hp: 2, rnn_dims_hp: 70, dnn_nums_hp: 2}, 'last')
