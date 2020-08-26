import io
import os
import itertools
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt, gridspec as gridspec
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tools.exec_tools import create_paths


def lnr_schedule(step):
    begin_rate = 0.001
    decay_rate = 0.7
    decay_step = 50

    learn_rate = begin_rate * np.power(decay_rate, np.divmod(step, decay_step)[0])
    tf.summary.scalar('Learning Rate', data=learn_rate, step=step)

    return learn_rate


def rop_schedule():
    rop_callback = ReduceLROnPlateau(
        monitor='val_loss',
        min_delta=0.001,
        factor=0.5,
        mode='min',
        patience=10,
        cooldown=5,
        verbose=1,
    )

    return rop_callback


def plot_confusion(matrix, report, categories):
    fig = plt.figure(figsize=(9, 12)); gs = gridspec.GridSpec(3, 2)
    ax1 = plt.subplot(gs[0:2, :]); ax2 = plt.subplot(gs[2, :])

    im = ax1.imshow(matrix, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xticks(np.arange(len(categories))); ax1.set_xticklabels(categories, rotation=45)
    ax1.set_yticks(np.arange(len(categories))); ax1.set_yticklabels(categories)

    threshold = 0.5 * matrix.max()
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        color = 'white' if matrix[i, j] > threshold else 'black'
        ax1.text(j, i, matrix[i, j], horizontalalignment='center', color=color)
        ax1.set_ylabel('True label')
        ax1.set_xlabel('Predicted label')

    ax2.set_title('Classification Report', position=(0.4, 1))
    ax2.text(0.35, 0.6, report, size=12, family='monospace', ha='center', va='center')
    ax2.axis('off')

    plt.tight_layout()

    return fig


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    img = tf.image.decode_png(buf.getvalue(), channels=4)
    img = tf.expand_dims(img, 0)

    return img