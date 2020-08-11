import numpy as np
import tensorflow as tf
from model._base import _Base, _FoldBase
from tensorflow.keras.layers import LSTM, Dense, ReLU
from tensorflow.keras.layers import LayerNormalization, GlobalAveragePooling1D, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from layer.phased import PhasedLSTM
from config.data_config import data_config
from config.model_config import rnn_nums_hp, rnn_dims_hp, dnn_nums_hp
from config.exec_config import train_config

use_gen = train_config['use_gen']
(w, s) = data_config['ws']


class SimpleLSTM(_Base):

    def __init__(self, dataset_name, model_name, hyper_param, exp_dir):
        super().__init__(dataset_name, model_name, hyper_param, exp_dir)

    def _build(self):
        model = Sequential()
        model.add(tf.keras.layers.Masking(mask_value=np.pi, dtype=np.float32, input_shape=(None, w * 2)))

        for _ in range(self.hyper_param[rnn_nums_hp]):
            # foo = True if _ < self.hyper_param[rnn_nums_hp] - 1 else False
            LSTMLayer = LSTM if self.model_name == 'sim' else PhasedLSTM
            model.add(LSTMLayer(
                units=self.hyper_param[rnn_dims_hp], return_sequences=True,
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

        model.add(GlobalAveragePooling1D(data_format='channels_last'))
        model.add(Dense(len(self.categories), activation='softmax'))

        self.model = model


class FoldSimpleLSTM(SimpleLSTM, _FoldBase):

    def __init__(self, dataset_name, model_name, hyper_param, exp_dir, fold):
        _FoldBase.__init__(self, dataset_name, model_name, hyper_param, exp_dir, fold)
