import io
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf


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
