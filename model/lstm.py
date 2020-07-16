import os
from global_settings import DATA_FOLDER
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tools.data_tools import cat_generator, cat_loader
from datetime import datetime

from config.model_confg import rnn_config

generator = rnn_config['generator']
rnn_num, rnn_dim = rnn_config['rnn_num'], rnn_config['rnn_dim']
dnn_num, drop = rnn_config['dnn_num'], rnn_config['drop']
epochs, batch_size = rnn_config['epochs'], rnn_config['batch_size']


def lstm(X_train, y_train):
    model = Sequential()
    model.add(GRU(units=rnn_dim, input_shape=[None, X_train.shape[2]]))

    for _ in range(rnn_num-1):
        model.add(GRU(units=rnn_dim))

    for _ in range(dnn_num):
        model.add(Dense(units=rnn_dim*2, activation='relu'))
        model.add(Dropout(rate=drop))

    model.add(Dense(y_train.shape[1], activation='softmax'))

    return model


def train_lstm(model, dataset):
    NAME = '-'.join([f'rnn_num_{rnn_num}', f'rnn_dim_{rnn_dim}', f'dnn_num_{dnn_num}'])
    log_dir = os.path.join(DATA_FOLDER, f'{dataset}_log')
    if not os.path.isdir(log_dir): os.mkdir(log_dir)
    log_path = os.path.join(log_dir, '-'.join([NAME, datetime.now().strftime("%Y%m%d-%H%M%S")]))
    callbacks = TensorBoard(log_dir=log_path)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(
        x=None, y=None, batch_size=batch_size, epochs=1, verbose=0,
        validation_data=(None, None), callbacks=[callbacks],
        max_queue_size=10, workers=5, use_multiprocessing=True
    )


def test_lstm(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 15, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)

    return accuracy
