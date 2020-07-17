import os
import itertools
import tensorflow as tf
from global_settings import DATA_FOLDER
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorboard.plugins.hparams import api as hp
from datetime import datetime
from config.train_config import train_config
from config.model_config import rnn_config
from tools.data_tools import data_loader, data_generator, w
from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp

generator, epoch = train_config['generator'], train_config['epoch']
batch, metric = train_config['batch'], train_config['metric']
drop = rnn_config['drop']


def loop_lstm(dataset_name):
    hyp_path = os.path.join(DATA_FOLDER, f'{dataset_name}_log', 'hyper_params')
    with tf.summary.create_file_writer(hyp_path).as_default():
        hp.hparams_config(
            hparams=[rnn_nums_hp, rnn_dims_hp, dnn_nums_hp],
            metrics=[hp.Metric(metric)],
        )

    rnn_nums, rnn_dims, dnn_nums = rnn_nums_hp.domain.values, rnn_dims_hp.domain.values, dnn_nums_hp.domain.values
    for rnn_num, rnn_dim, dnn_num in itertools.product(rnn_nums, rnn_dims, dnn_nums):
        hyper_params = {rnn_nums_hp: rnn_num, rnn_dims_hp: rnn_dim, dnn_nums_hp: dnn_num}
        log_lstm(dataset_name, hyper_params)


def log_lstm(dataset_name, hyper_params):
    rnn_num, rnn_dim, dnn_num = hyper_params[rnn_nums_hp], hyper_params[rnn_dims_hp], hyper_params[dnn_nums_hp]
    hyp_dir = os.path.join(DATA_FOLDER, f'{dataset_name}_log', 'hyper_params')
    if not os.path.isdir(hyp_dir): os.mkdir(hyp_dir)
    log_name = '-'.join([f'rnn_num_{rnn_num}', f'rnn_dim_{rnn_dim}', f'dnn_num_{dnn_num}'])
    log_path = os.path.join(hyp_dir, '-'.join([log_name, datetime.now().strftime('%Y%m%d-%H%M%S')]))

    print('--- Starting trial: %s' % log_name)
    print({h.name: hyper_params[h] for h in hyper_params})

    run_lstm(dataset_name, hyper_params, log_path)


def run_lstm(dataset_name, hyper_params, log_path):
    metric_callbacks = TensorBoard(log_dir=log_path)
    hypers_callbacks = hp.KerasCallback(writer=log_path, hparams=hyper_params)
    model = build_lstm(hyper_params)

    x_valid, y_valid = data_loader(dataset_name, 'valid')
    if generator:
        zip_train = data_generator(dataset_name, 'train')
        model.fit(
            x=zip_train, epochs=epoch, verbose=0,
            validation_data=(x_valid, y_valid), callbacks=[metric_callbacks, hypers_callbacks],
            max_queue_size=10, workers=5, use_multiprocessing=False
        )
    else:
        x_train, y_train = data_loader(dataset_name, 'train')
        model.fit(
            x=x_train, y=y_train, batch_size=batch, epochs=epoch, verbose=0,
            validation_data=(x_valid, y_valid), callbacks=[metric_callbacks, hypers_callbacks]
        )


def build_lstm(hyper_params):
    model = Sequential()
    model.add(GRU(units=hyper_params[rnn_dims_hp], input_shape=[None, w * 2]))

    for _ in range(hyper_params[rnn_nums_hp]-1):
        model.add(GRU(units=hyper_params[rnn_dims_hp]))

    for _ in range(hyper_params[dnn_nums_hp]):
        model.add(Dense(units=hyper_params[rnn_dims_hp]*2, activation='relu'))
        model.add(Dropout(rate=drop))

    model.add(Dense(1, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=[metric])

    return model
