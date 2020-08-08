import numpy as np
import tensorflow as tf
from model.base import Base, FoldBase
from tensorflow.keras.layers import LSTM, Dense, ReLU
from tensorflow.keras.layers import LayerNormalization, GlobalAveragePooling1D, TimeDistributed
from tensorflow.keras.models import Sequential
from tools.utils import load_one_hot
from tensorflow.keras import regularizers
from config.data_config import data_config
from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp
from tools.data_tools import data_loader, DataGenerator
from tools.data_tools import fold_loader, FoldGenerator
from config.exec_config import train_config

use_gen = train_config['use_gen']
(w, s) = data_config['ws']


class SimpleLSTM(Base):

    def __init__(self, dataset_name, hyper_param, exp_dir):
        super().__init__(dataset_name, hyper_param, exp_dir)

    def _load_data(self):
        self.train = data_loader(self.dataset_name, 'sim', 'train') if not use_gen else DataGenerator(self.dataset_name)
        self.x_valid, self.y_valid = data_loader(self.dataset_name, 'sim', 'valid')
        self.x_evalu, self.y_evalu = data_loader(self.dataset_name, 'sim', 'evalu')

    def _build(self):
        model = Sequential()
        model.add(tf.keras.layers.Masking(mask_value=np.pi, dtype=np.float32, input_shape=(None, w * 2)))

        for _ in range(self.hyper_param[rnn_nums_hp]):
            # rnn_num = self.hyper_param[rnn_nums_hp]
            # foo = True if _ < rnn_num - 1 else False
            foo = True
            model.add(LSTM(
                units=self.hyper_param[rnn_dims_hp], return_sequences=foo,
                recurrent_regularizer=regularizers.l2(0.1))
            )
            model.add(LayerNormalization())

        for _ in range(self.hyper_param[dnn_nums_hp]):
            model.add(TimeDistributed(Dense(
                units=self.hyper_param[rnn_dims_hp] * 2,
                kernel_regularizer=regularizers.l2(0.1)))
            )
            model.add(LayerNormalization())
            model.add(ReLU())

        encoder = load_one_hot(self.dataset_name)
        model.add(GlobalAveragePooling1D(data_format='channels_last'))
        model.add(Dense(len(encoder.categories_[0]), activation='softmax'))

        self.model = model


class FoldSimpleLSTM(SimpleLSTM, FoldBase):

    def __init__(self, dataset_name, hyper_param, exp_dir, fold):
        FoldBase.__init__(self, dataset_name, hyper_param, exp_dir, fold)

    def _load_data(self):
        if not use_gen:
            self.train = fold_loader(self.dataset_name, 'sim', 'train', self.fold)
        else:
            self.train = FoldGenerator(self.dataset_name, self.fold)
        self.x_evalu, self.y_evalu = fold_loader(self.dataset_name, 'sim', 'evalu', self.fold)
        self.x_valid, self.y_valid = self.x_evalu.copy(), self.y_evalu.copy()
