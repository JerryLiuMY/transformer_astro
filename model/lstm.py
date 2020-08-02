import numpy as np
import tensorflow as tf
from model.base import Base, FoldBase
from tensorflow.keras.layers import LSTM, Dense, ReLU, Dropout
from tensorflow.keras.layers import LayerNormalization, GlobalAveragePooling1D, TimeDistributed
from tensorflow.keras.models import Sequential
from tools.utils import load_one_hot
from tensorflow.keras import regularizers
from config.data_config import data_config
from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp

window = data_config['window']


class SimpleLSTM(Base):

    def __init__(self, dataset_name, hyper_param, exp_dir):
        super().__init__(dataset_name, hyper_param, exp_dir)

    def build(self):
        model = Sequential()
        model.add(tf.keras.layers.Masking(mask_value=-10, dtype=np.float32, input_shape=(None, window * 2)))

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


class FoldBasic(SimpleLSTM, FoldBase):

    def __init__(self, dataset_name, hyper_param, exp_dir, fold):
        FoldBase.__init__(self, dataset_name, hyper_param, exp_dir, fold)
