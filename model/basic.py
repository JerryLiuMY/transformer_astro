import os
import itertools
import numpy as np
import tensorflow as tf
from model.base import Base
from global_settings import DATA_FOLDER
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.models import Sequential
from tools.utils import load_one_hot, new_dir
from config.data_config import data_config
from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp
from config.train_config import train_config
from model.base import log_params

window = data_config['window']
generator, epoch = train_config['generator'], train_config['epoch']
batch, metrics = train_config['batch'], train_config['metrics']
metric_names = ['epoch_loss'] + ['_'.join(['epoch', _.name]) for _ in metrics]


class Basic(Base):

    def __init__(self, dataset_name, hyper_param, exp_dir):
        super().__init__(dataset_name, hyper_param, exp_dir)

    def build(self):
        model = Sequential()
        model.add(tf.keras.layers.Masking(mask_value=0.0, dtype=np.float32, input_shape=(None, window * 2)))

        for _ in range(self.hyper_param[rnn_nums_hp]):
            foo = True if _ == 0 and self.hyper_param[rnn_nums_hp] >= 2 else False
            model.add(GRU(units=self.hyper_param[rnn_dims_hp], return_sequences=foo))

        for _ in range(self.hyper_param[dnn_nums_hp]):
            model.add(Dense(units=self.hyper_param[rnn_dims_hp] * 2, activation='tanh'))
            model.add(Dropout(rate=0.4))

        encoder = load_one_hot(self.dataset_name)
        model.add(Dense(len(encoder.categories_[0]), activation='softmax'))

        self.model = model


def run(dataset_name):
    exp_dir = new_dir(os.path.join(DATA_FOLDER, f'{dataset_name}_log'))
    log_params(exp_dir)
    rnn_nums, rnn_dims, dnn_nums = rnn_nums_hp.domain.values, rnn_dims_hp.domain.values, dnn_nums_hp.domain.values
    for rnn_num, rnn_dim, dnn_num in itertools.product(rnn_nums, rnn_dims, dnn_nums):
        hyper_param = {rnn_nums_hp: rnn_num, rnn_dims_hp: rnn_dim, dnn_nums_hp: dnn_num}
        exp = Basic(dataset_name=dataset_name, hyper_param=hyper_param, exp_dir=exp_dir)
        exp.build()
        exp.train()


if __name__ == '__main__':
    run('ASAS')
