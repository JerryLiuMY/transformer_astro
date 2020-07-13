import os
import numpy as np
import pandas as pd
from global_settings import DATA_FOLDER
w, s = 30, 20


def cat_loader(dataset, file):
    assert dataset in ['ASAS', 'MACHO']
    catalog = pd.read_csv(os.path.join(DATA_FOLDER, 'ASAS', 'catalog.csv'), index_col=0)
    for cat, path in list(zip(catalog['Class'], catalog['Path'])):
        df = pd.read_csv(os.path.join(DATA_FOLDER, 'ASAS', path))
        df.sort_values(by=['mjd'], inplace=True).reset_index(drop=True, inplace=True)

    yield dtdm_bin


def binning(df):
    mjd, mag = np.diff(df['mjd'].values).reshape(-1, 1), np.diff(df['mag'].values).reshape(-1, 1)
    dtdm_org = np.concatenate([mjd, mag], axis=1)
    dtdm_bin = np.array([], dtype=np.int64).reshape(0, 2 * w)
    for i in np.arange(np.shape(dtdm_org)[0] - w + 1)[::s]:
        dtdm_bin = np.vstack([dtdm_bin, dtdm_org[i: i + w, :].reshape(-1)])



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
