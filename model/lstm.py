import os
import itertools
import numpy as np
import tensorflow as tf
from global_settings import DATA_FOLDER
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback
from tensorboard.plugins.hparams.api import KerasCallback
from tensorboard.plugins.hparams import api as hp
from datetime import datetime
from config.train_config import train_config
from config.model_config import rnn_config
from tools.data_tools import data_loader, data_generator, load_one_hot, window
from tools.model_tools import log_confusion
from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp

generator, epoch = train_config['generator'], train_config['epoch']
batch, metrics = train_config['batch'], train_config['metrics']
drop = rnn_config['drop']


def run_lstm(dataset_name):
    log_dir = os.path.join(DATA_FOLDER, f'{dataset_name}_log')
    if not os.path.isdir(log_dir): os.mkdir(log_dir)
    hyp_path = os.path.join(log_dir, 'hyper_params')
    with tf.summary.create_file_writer(hyp_path).as_default():
        hp.hparams_config(
            hparams=[rnn_nums_hp, rnn_dims_hp, dnn_nums_hp],
            metrics=[hp.Metric('_'.join(['epoch', _.name])) for _ in metrics],
        )

    rnn_nums, rnn_dims, dnn_nums = rnn_nums_hp.domain.values, rnn_dims_hp.domain.values, dnn_nums_hp.domain.values
    for rnn_num, rnn_dim, dnn_num in itertools.product(rnn_nums, rnn_dims, dnn_nums):
        hyper_params = {rnn_nums_hp: rnn_num, rnn_dims_hp: rnn_dim, dnn_nums_hp: dnn_num}
        exp_path = lstm_log(dataset_name, hyper_params)
        lstm_step(dataset_name, hyper_params, exp_path)


def lstm_log(dataset_name, hyper_params):
    rnn_num, rnn_dim, dnn_num = hyper_params[rnn_nums_hp], hyper_params[rnn_dims_hp], hyper_params[dnn_nums_hp]
    log_dir = os.path.join(DATA_FOLDER, f'{dataset_name}_log')
    exp_name = '-'.join([f'rnn_num_{rnn_num}', f'rnn_dim_{rnn_dim}', f'dnn_num_{dnn_num}'])
    exp_path = os.path.join(log_dir, '-'.join([exp_name, datetime.now().strftime('%Y%m%d-%H%M%S')]))

    print(f'--- Starting trial: {exp_name}')
    print({h.name: hyper_params[h] for h in hyper_params})

    return exp_path


def lstm_step(dataset_name, hyper_params, exp_path):
    metric_callbacks = TensorBoard(log_dir=exp_path)
    hypers_callbacks = KerasCallback(writer=exp_path, hparams=hyper_params)
    model = lstm(dataset_name, hyper_params)
    callbacks = [metric_callbacks, hypers_callbacks]

    if generator:
        zip_train = data_generator(dataset_name, 'train')
        x_valid, y_valid = data_loader(dataset_name, 'valid')
        model.fit(
            x=zip_train, epochs=epoch, verbose=1,
            validation_data=(x_valid, y_valid), callbacks=callbacks,
            max_queue_size=10, workers=5, use_multiprocessing=False
        )
    else:
        x_train, y_train = data_loader(dataset_name, 'train')
        x_valid, y_valid = data_loader(dataset_name, 'valid')
        model.fit(
            x=x_train, y=y_train, batch_size=batch, epochs=epoch, verbose=1,
            validation_data=(x_valid, y_valid), callbacks=callbacks
        )

    # confusion_callback = LambdaCallback(on_epoch_end=log_confusion)


def lstm(dataset_name, hyper_params):
    model = Sequential()
    model.add(tf.keras.layers.Masking(mask_value=0.0, dtype=np.float32, input_shape=(None, window * 2)))
    model.add(GRU(units=hyper_params[rnn_dims_hp]))

    for _ in range(hyper_params[rnn_nums_hp]-1):
        model.add(GRU(units=hyper_params[rnn_dims_hp]))

    for _ in range(hyper_params[dnn_nums_hp]):
        model.add(Dense(units=hyper_params[rnn_dims_hp]*2, activation='tanh'))
        model.add(Dropout(rate=drop))

    encoder = load_one_hot(dataset_name)
    model.add(Dense(len(encoder.categories_[0]), activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=metrics)

    return model

# confusion matrix
# F1 score
# loop 10 times
# Stop training after loss stabilize
# k-fold validation
# attention model
# Phased LSTM

