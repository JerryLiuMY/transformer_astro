import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
from tools.log_tools import plot_confusion, plot_timesteps
from tools.exec_tools import create_dirs
import json


def evaluate(exp, y_evalu, y_predi_seq):
    y_evalu, y_predi = y_evalu, y_predi_seq[:, -1, :]
    exp_dir, exp_name = exp.exp_dir, exp.exp_name
    eva_dir = os.path.join(exp_dir, 'test'); create_dirs(eva_dir)
    eva_path = os.path.join(eva_dir, exp_name); create_dirs(eva_path)

    y_evalu_spar = exp.encoder.inverse_transform(y_evalu)
    y_predi_spar = exp.encoder.inverse_transform(y_predi)
    confusion_fig = plot_confusion(y_evalu_spar, y_predi_spar, categories=exp.categories)
    timesteps_fig = plot_timesteps(y_evalu, y_predi_seq)
    confusion_fig.savefig(os.path.join(eva_path, 'confusion.pdf'), bbox_inches='tight')
    timesteps_fig.savefig(os.path.join(eva_path, 'timesteps.pdf'), bbox_inches='tight')

    ccl = tf.keras.losses.CategoricalCrossentropy()
    ccm = tf.keras.metrics.CategoricalAccuracy(); ccm.update_state(y_evalu, y_predi)
    pre = tf.keras.metrics.Precision(); pre.update_state(y_evalu, y_predi)
    rec = tf.keras.metrics.Recall(); rec.update_state(y_evalu, y_predi)

    summary = {
        "categorical_crossentropy_loss": float(ccl(y_evalu, y_predi).numpy()),
        "categorical_accuracy": float(ccm.result().numpy()),
        "precision": float(pre.result().numpy()),
        "recall": float(rec.result().numpy())
    }

    with open(os.path.join(eva_path, 'summary.json'), "w", encoding="utf8") as f:
        json.dump(summary, f)


def predict(exp):
    model = Model(inputs=exp.model.inputs, outputs=exp.model.get_layer('softmax').output)
    y_evalu = np.array([]).reshape(0, len(exp.categories))
    for x_evalu_, y_evalu_ in exp.dataset_evalu.take(-1):
        y_evalu = np.vstack([y_evalu, y_evalu_.numpy()])

    with tf.device('/CPU:0'):
        y_somax_seq = model.predict(exp.dataset_evalu)
        y_predi_seq = np.zeros(shape=np.shape(y_somax_seq))
        for t in range(np.shape(y_somax_seq)[1]):
            max_arg = tf.math.argmax(y_somax_seq[:, t, :], axis=1)
            y_predi = tf.one_hot(max_arg, depth=len(exp.categories)).numpy()
            y_predi_seq[:, t, :] = y_predi

    return y_evalu, y_predi_seq


if __name__ == '__main__':
    from tools.test_tools import get_exp
    from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp
    hyper_param = {rnn_nums_hp: 2, rnn_dims_hp: 70, dnn_nums_hp: 2}
    exp = get_exp('ASAS', 'sim', '10', hyper_param, 'last')
    y_evalu, y_predi_seq = predict(exp)
    evaluate(exp, y_evalu, y_predi_seq)
