import os
import json
import keras
from tf.keras.layers import Bidirectional, LSTM, Dropout, Dense
from tf.keras.models import Sequential
from global_settings import CONFIG_FOLDER
from tools.data_tools import cat_generator, cat_loader

from config.model_confg import rnn_config

rnn_config


def run_lstm(model, dataset):

    model.fit(
        x=None, y=None, epochs=1, verbose=2, callbacks=None,
        max_queue_size=10, workers=1, use_multiprocessing=True
    )


def lstm(X_train, y_train):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=128, input_shape=[None, X_train.shape[2]])))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model



model = Sequential()

model.add(LSTM(32, return_sequences=True, input_shape=(None, 5)))
model.add(LSTM(8, return_sequences=True))
model.add(TimeDistributed(Dense(2, activation='sigmoid')))

print(model.summary(90))

model.compile(loss='categorical_crossentropy',
              optimizer='adam')
