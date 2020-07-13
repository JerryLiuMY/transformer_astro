from keras.layers import Bidirectional, LSTM, Dropout, Dense
from global_settings import CONFIGS_FOLDER
from tools.data_tools import cat_generator, cat_loader
import os
import json
import keras


with open(os.path.join(CONFIGS_FOLDER, 'portfolio.json'), 'rb') as handle:
    params = json.load(handle)
    generator = params["generator"]


def run_lstm(model, dataset):
    if generator:
        history = model.fit_generator(cat_generator(dataset))
    else:
        X, y = cat_loader(dataset)
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



from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed

model = Sequential()

model.add(LSTM(32, return_sequences=True, input_shape=(None, 5)))
model.add(LSTM(8, return_sequences=True))
model.add(TimeDistributed(Dense(2, activation='sigmoid')))

print(model.summary(90))

model.compile(loss='categorical_crossentropy',
              optimizer='adam')

model.fit_generator(train_generator(), steps_per_epoch=30, epochs=10, verbose=1)
