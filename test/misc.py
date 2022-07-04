import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from test.evaluation import predict
from tools.data_tools import load_catalog, load_sliding
from datetime import datetime


def plot_timesteps(exp):
    # evaluate accuracy
    print(f'{datetime.now()} Plotting time step figure')
    y_evalu, y_predi_seq = predict(exp)
    acc_seq, metric = np.array([]), tf.keras.metrics.CategoricalAccuracy()
    for step in range(np.shape(y_predi_seq)[1]):
        metric.update_state(y_evalu, y_predi_seq[:, step, :])
        acc_seq = np.append(acc_seq, metric.result().numpy())

    # plot
    with sns.axes_style("darkgrid"):
        timesteps_fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(acc_seq)
        ax.set_xlabel('time steps')
        ax.set_ylabel('test categorical accuracy')
        plt.show()

    return timesteps_fig


def count_drop(dataset_name, set_type):
    if set_type == 'evalu':
        catalog_len = len(set(load_catalog(dataset_name, 'analy')['Path']))
    else:
        catalog_len = len(set(load_catalog(dataset_name, set_type)['Path']))
    sliding_len = len(set(load_sliding(dataset_name, set_type)['Path']))
    print(f'{dataset_name} {set_type} set: {catalog_len - sliding_len} lightcurves dropped out of {catalog_len}')
