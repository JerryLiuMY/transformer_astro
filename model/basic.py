import numpy as np
import tensorflow as tf
from model.base import Base, FoldBase
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.models import Sequential
from tools.utils import load_one_hot
from config.data_config import data_config
from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp
from config.exec_config import train_config

window = data_config['window']
use_gen, epoch = train_config['use_gen'], train_config['epoch']
batch, metrics = train_config['batch'], train_config['metrics']
metric_names = ['epoch_loss'] + ['_'.join(['epoch', _.name]) for _ in metrics]


class Basic(Base):

    def __init__(self, dataset_name, hyper_param, exp_dir):
        super().__init__(dataset_name, hyper_param, exp_dir)

    def build(self):
        self.model = build(self.dataset_name, self.hyper_param)


class FoldBasic(FoldBase):

    def __init__(self, dataset_name, hyper_param, exp_dir, fold):
        super().__init__(dataset_name, hyper_param, exp_dir, fold)

    def build(self):
        self.model = build(self.dataset_name, self.hyper_param)


def build(dataset_name, hyper_param):
    model = Sequential()
    model.add(tf.keras.layers.Masking(mask_value=0.0, dtype=np.float32, input_shape=(None, window * 2)))

    for _ in range(hyper_param[rnn_nums_hp]):
        foo = True if _ == 0 and hyper_param[rnn_nums_hp] >= 2 else False
        model.add(GRU(units=hyper_param[rnn_dims_hp], return_sequences=foo))

    for _ in range(hyper_param[dnn_nums_hp]):
        model.add(Dense(units=hyper_param[rnn_dims_hp] * 2, activation='tanh'))
        model.add(Dropout(rate=0.4))

    encoder = load_one_hot(dataset_name)
    model.add(Dense(len(encoder.categories_[0]), activation='softmax'))

    return model

