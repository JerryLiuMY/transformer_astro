import os
import itertools
from global_settings import DATA_FOLDER
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
from config.train_config import train_config
from config.model_confg import base
from tools.data_tools import data_loader, data_generator, w

generator, drop = base['generator'], base['drop']
epoch, batch = train_config['epoch'], train_config['batch']

rnn_nums = [1, 2]
rnn_dims = [25, 50, 100, 200]
dnn_nums = [1, 2]


def run_lstm(dataset_name):
    for values in itertools.product(rnn_nums, rnn_dims, dnn_nums):
        rnn_config = base.copy()
        rnn_num, rnn_dim, dnn_num = values
        rnn_config['rnn_num'] = rnn_num
        rnn_config['rnn_dim'] = rnn_dim
        rnn_config['dnn_num'] = dnn_num

        model = lstm(rnn_config)
        history = train_lstm(dataset_name, model, rnn_config)


def lstm(config):
    rnn_num = config['rnn_num']
    rnn_dim = config['rnn_dim']
    dnn_num = config['dnn_num']

    model = Sequential()
    model.add(GRU(units=rnn_dim, input_shape=[None, w * 2]))

    for _ in range(rnn_num-1):
        model.add(GRU(units=rnn_dim))

    for _ in range(dnn_num):
        model.add(Dense(units=rnn_dim*2, activation='relu'))
        model.add(Dropout(rate=drop))

    model.add(Dense(1, activation='softmax'))

    return model


def train_lstm(dataset_name, model, config):
    rnn_num, rnn_dim, dnn_num = config['rnn_num'], config['rnn_dim'], config['dnn_num']
    log_dir = os.path.join(DATA_FOLDER, f'{dataset_name}_log')
    if not os.path.isdir(log_dir): os.mkdir(log_dir)
    NAME = '-'.join([f'rnn_num_{rnn_num}', f'rnn_dim_{rnn_dim}', f'dnn_num_{dnn_num}'])
    log_path = os.path.join(log_dir, '-'.join([NAME, datetime.now().strftime('%Y%m%d-%H%M%S')]))

    callbacks = TensorBoard(log_dir=log_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    x_train, y_train = data_generator(dataset_name, 'train') if generator else data_loader(dataset_name, 'train')
    x_valid, y_valid = data_loader(dataset_name, 'valid')

    history = model.fit(
        x=x_train, y=y_train, batch_size=batch, epochs=epoch, verbose=0,
        validation_data=(x_valid, y_valid), callbacks=[callbacks],
        max_queue_size=10, workers=5, use_multiprocessing=True
    )

    return history


def test_lstm(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 15, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)

    return accuracy
