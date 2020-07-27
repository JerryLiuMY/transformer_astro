import numpy as np
import tensorflow as tf
from model.base import Base, FoldBase
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.models import Sequential
from tools.utils import load_one_hot
from tensorflow.keras import regularizers
from config.data_config import data_config
from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp

window = data_config['window']


class SimpleGRU(Base):

    def __init__(self, dataset_name, hyper_param, exp_dir):
        super().__init__(dataset_name, hyper_param, exp_dir)

    def build(self):
        model = Sequential()
        model.add(tf.keras.layers.Masking(mask_value=0.0, dtype=np.float32, input_shape=(None, window * 2)))

        for _ in range(self.hyper_param[rnn_nums_hp]):
            foo = True if _ == 0 and self.hyper_param[rnn_nums_hp] >= 2 else False
            model.add(GRU(units=self.hyper_param[rnn_dims_hp], return_sequences=foo,
                          recurrent_regularizer=regularizers.l2(0.05)))

        for _ in range(self.hyper_param[dnn_nums_hp]):
            model.add(Dense(units=self.hyper_param[rnn_dims_hp] * 2, activation='tanh',
                            kernel_regularizer=regularizers.l2(0.05)))
            model.add(Dropout(rate=0.2))

        encoder = load_one_hot(self.dataset_name)
        model.add(Dense(len(encoder.categories_[0]), activation='softmax'))

        self.model = model


class FoldBasic(SimpleGRU, FoldBase):

    def __init__(self, dataset_name, hyper_param, exp_dir, fold):
        FoldBase.__init__(self, dataset_name, hyper_param, exp_dir, fold)
