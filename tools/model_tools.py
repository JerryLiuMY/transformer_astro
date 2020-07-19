import io
import seaborn
import itertools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
from tools.data_tools import data_loader, load_one_hot
import sklearn
seaborn.set()


logdir = 'logs/image/' + datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = TensorBoard(log_dir=logdir)


def log_confusion(epoch, logs):
    encoder = load_one_hot(dataset_name)
    categories = encoder.categories_[0]
    x_evalu, y_true = data_loader(dataset_name, 'valid')
    y_pred = tf.one_hot(tf.argmax(model.predict(x_evalu), dimension=1), depth=2)
    y_true_spar, y_pred_spar = encoder.inverse_transform(y_true), encoder.inverse_transform(y_pred)

    matrix = sklearn.metrics.confusion_matrix(y_true_spar, y_pred_spar)
    confusion_figure = plot_confusion(matrix, categories=categories)
    confusion_image = plot_to_image(confusion_figure)

    with tf.summary.create_file_writer(logdir + '/cm').as_default():
        tf.summary.image('Confusion Matrix', confusion_image, step=epoch)


def plot_confusion(matrix, categories):
    figure = plt.figure(figsize=(8, 8))
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
    """Converts the matplotlib plot specified by 'figure' to a PNG image and returns it."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image
