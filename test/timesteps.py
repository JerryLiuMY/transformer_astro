from datetime import datetime
import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
from test.evaluation import predict


def plot_timesteps(exp):
    # evaluate accuracy
    print(f'{datetime.now()} Plotting time step figure')
    y_evalu, y_predi_seq = predict(exp)
    acc_seq, metric = np.array([]), tf.keras.metrics.CategoricalAccuracy()
    for step in tqdm_notebook(range(np.shape(y_predi_seq)[1])):
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
