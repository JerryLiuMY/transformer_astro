from keras.layers import Bidirectional, LSTM, Dropout, Dense
from global_settings import CONFIGS_FOLDER
import os
import json
import keras


with open(os.path.join(CONFIGS_FOLDER, 'portfolio.json'), 'rb') as handle:
    params = json.load(handle)


def run_lstm(model, X_train, y_train):
    history = model.fit(X_train, y_train,
                        epochs=params['epochs'],
                        batch_size=params['batch_sze'],
                        validation_split=0.1, shuffle=True)

    return history


def lstm(X_train, y_train):
    model = keras.Sequential()
    model.add(Bidirectional(LSTM(units=128, input_shape=[None, X_train.shape[2]])))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model