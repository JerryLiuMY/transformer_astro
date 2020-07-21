from model.base import Base
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.models import Sequential
from tools.data_tools import load_one_hot, window
from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp
from config.train_config import train_config
from model.base import log_params

generator, epoch = train_config['generator'], train_config['epoch']
batch, metrics = train_config['batch'], train_config['metrics']
metric_names = ['epoch_loss'] + ['_'.join(['epoch', _.name]) for _ in metrics]


class Basic(Base):

    def __init__(self, dataset_name, hyper_params):
        super().__init__(dataset_name, hyper_params)

    def build(self):
        model = Sequential()
        model.add(tf.keras.layers.Masking(mask_value=0.0, dtype=np.float32, input_shape=(None, window * 2)))

        for _ in range(self.hyper_params[rnn_nums_hp]):
            foo = True if _ == 0 and self.hyper_params[rnn_nums_hp] >= 2 else False
            model.add(GRU(units=self.hyper_params[rnn_dims_hp], return_sequences=foo))

        for _ in range(self.hyper_params[dnn_nums_hp]):
            model.add(Dense(units=self.hyper_params[rnn_dims_hp]*2, activation='tanh'))
            model.add(Dropout(rate=0.4))

        encoder = load_one_hot(self.dataset_name)
        model.add(Dense(len(encoder.categories_[0]), activation='softmax'))

        self.model = model


def run(dataset_name):
    log_params(dataset_name)
    rnn_nums, rnn_dims, dnn_nums = rnn_nums_hp.domain.values, rnn_dims_hp.domain.values, dnn_nums_hp.domain.values
    for rnn_num, rnn_dim, dnn_num in itertools.product(rnn_nums, rnn_dims, dnn_nums):
        hyper_params = {rnn_nums_hp: rnn_num, rnn_dims_hp: rnn_dim, dnn_nums_hp: dnn_num}
        exp = Basic(dataset_name=dataset_name, hyper_params=hyper_params)
        exp.build()
        exp.train()


if __name__ == '__main__':
    dataset_name = 'MACHO'
    run(dataset_name)
