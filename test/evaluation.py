import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from data.loader import token_loader
from tensorflow.python.keras import Model
from tools.log_tools import plot_confusion
from tools.dir_tools import create_dirs
from data.loader import data_loader


def evaluate(exp):
    true, pred_seq = predict(exp)
    true, pred = true, pred_seq[:, -1, :]
    exp_dir, exp_name = exp.exp_dir, exp.exp_name
    eva_dir = os.path.join(exp_dir, 'test'); create_dirs(eva_dir)
    eva_path = os.path.join(eva_dir, exp_name); create_dirs(eva_path)

    true_spar = exp.encoder.inverse_transform(true)
    pred_spar = exp.encoder.inverse_transform(pred)
    confusion_fig = plot_confusion(true_spar, pred_spar, categories=exp.categories)
    confusion_fig.savefig(os.path.join(eva_path, 'confusion.pdf'), bbox_inches='tight')

    ccl = tf.keras.losses.CategoricalCrossentropy()
    ccm = tf.keras.metrics.CategoricalAccuracy(); ccm.update_state(true, pred)
    pre = tf.keras.metrics.Precision(); pre.update_state(true, pred)
    rec = tf.keras.metrics.Recall(); rec.update_state(true, pred)

    summary = {
        "categorical_crossentropy_loss": round(float(ccl(true, pred).numpy()), 3),
        "categorical_accuracy": round(float(ccm.result().numpy()), 3),
        "precision": round(float(pre.result().numpy()), 3),
        "recall": round(float(rec.result().numpy()), 3)
    }

    with open(os.path.join(eva_path, 'summary.json'), "w", encoding="utf8") as f:
        json.dump(summary, f)


def predict(exp):
    token, dataset_evalu = token_loader(exp.dataset_name), data_loader(exp.dataset_name, 'evalu')
    with tf.device('/CPU:0'):
        # generate prediction
        model = Model(inputs=exp.model.inputs, outputs=exp.model.get_layer('softmax').output)
        y_evalu = np.array([]).reshape((0, len(exp.categories)))
        for x_evalu_, y_evalu_ in dataset_evalu.take(-1):
            y_evalu = np.vstack([y_evalu, y_evalu_.numpy()])
        y_somax_seq = model.predict(dataset_evalu)

        # group prediction
        true = np.array([]).reshape((0, len(exp.categories)))
        soft_seq = np.array([]).reshape((0, np.shape(y_somax_seq)[1], np.shape(y_somax_seq)[2]))
        for tok in pd.unique(token):
            index = np.array([idx for idx, foo in enumerate(token) if foo == tok])
            true = np.vstack([true, np.mean(y_evalu[index], axis=0, keepdims=True)])
            soft_seq = np.vstack([soft_seq, np.mean(y_somax_seq[index], axis=0, keepdims=True)])

        # convert to one-hot
        pred_seq = np.zeros(shape=np.shape(soft_seq))
        for t in range(np.shape(soft_seq)[1]):
            max_arg = tf.math.argmax(soft_seq[:, t, :], axis=1)
            y_predi = tf.one_hot(max_arg, depth=len(exp.categories)).numpy()
            pred_seq[:, t, :] = y_predi

    return true, pred_seq
