import io
import itertools
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt, gridspec as gridspec
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report


def lnr_schedule(step):
    begin_rate = 0.001
    decay_rate = 0.7
    decay_step = 30

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


def plot_confusion(y_evalu_spar, y_predi_spar, categories):
    # setting up
    matrix = np.around(confusion_matrix(y_evalu_spar, y_predi_spar, labels=categories), decimals=2)
    report = classification_report(y_evalu_spar, y_predi_spar, labels=categories, zero_division=0)
    confusion_fig, gs = plt.figure(figsize=(9, 12)), gridspec.GridSpec(3, 2)
    ax1, ax2 = plt.subplot(gs[0:2, :]), plt.subplot(gs[2, :])

    # confusion matrix
    im = ax1.imshow(matrix, interpolation='nearest', cmap='Blues')
    confusion_fig.colorbar(im, ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xticks(np.arange(len(categories))); ax1.set_xticklabels(categories, rotation=45)
    ax1.set_yticks(np.arange(len(categories))); ax1.set_yticklabels(categories)

    threshold = np.amax(0.5 * matrix)
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        color = 'white' if matrix[i, j] > threshold else 'black'
        ax1.text(j, i, matrix[i, j], horizontalalignment='center', color=color)
        ax1.set_ylabel('True label')
        ax1.set_xlabel('Predicted label')

    # classification report
    ax2.set_title('Classification Report', position=(0.4, 1))
    ax2.text(0.35, 0.6, report, size=12, family='monospace', ha='center', va='center')
    ax2.axis('off')

    # clean up
    confusion_fig.tight_layout()

    return confusion_fig


def fig_to_img(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    confusion_img = tf.image.decode_png(buf.getvalue(), channels=4)
    confusion_img = tf.expand_dims(confusion_img, 0)

    return confusion_img

