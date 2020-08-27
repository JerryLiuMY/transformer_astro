import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
from datetime import datetime
from test.prediction import predict, get_model
from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp


def plot_timestep(y_evalu, y_predi_seq):
    # evaluate accuracy
    print(f'{datetime.now()} Plotting time step figure')
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


def evaluate(exp, y_evalu, y_predi_seq):
    y_evalu_spar = exp.encoder.inverse_transform(y_evalu)
    y_predi_spar = exp.encoder.inverse_transform(y_predi_seq[:, -1, :])
    y_predi = y_predi_seq[:, -1, :]


# if __name__ == '__main__':
#     hyper_param = {rnn_nums_hp: 2, rnn_dims_hp: 70, dnn_nums_hp: 2}
#     exp = get_model('ASAS', 'sim', '10', hyper_param, 'last')
#     y_evalu, y_predi_seq = predict(exp)
#     plot_timestep(y_evalu, y_predi_seq)
#     evaluate(exp, y_evalu, y_predi_seq)
