import io
import itertools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_confusion(matrix, categories):
    figure = plt.figure(figsize=(10, 10))
    plt.imshow(matrix, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(categories))
    plt.xticks(tick_marks, categories, rotation=45)
    plt.yticks(tick_marks, categories)

    threshold = 0.5 * matrix.max()
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        color = 'white' if matrix[i, j] > threshold else 'black'
        plt.text(j, i, matrix[i, j], horizontalalignment='center', color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    return figure


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image
