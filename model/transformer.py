from model._base import _Base, _FoldBase
from tensorflow.keras.layers import GlobalAveragePooling1D, Input, Dense
from layer.encoder import Embedding, Encoder
from config.model_config import heads_hp, emb_dims_hp, ffn_dims_hp
from config.data_config import data_config
from config.exec_config import train_config
from keras import Model

window = data_config['window']
use_gen, ws = train_config['use_gen'], data_config['ws']


class Transformer(_Base):

    def __init__(self, dataset_name, hyper_param, exp_dir):
        super().__init__(dataset_name, hyper_param, exp_dir)

    def _build(self):
        (w, s) = ws[self.dataset_name]; seq_len = (window[self.dataset_name] - w) // s + 1
        inputs = Input(shape=(seq_len, w * 2))
        embedding = Embedding(seq_len, self.hyper_param[emb_dims_hp])
        transformer = Encoder(self.hyper_param[heads_hp],
                              self.hyper_param[emb_dims_hp],
                              self.hyper_param[ffn_dims_hp])

        x = embedding(inputs)
        x = transformer(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(self.hyper_param[ffn_dims_hp], activation="relu")(x)
        outputs = Dense(units=len(self.categories), activation="softmax", name='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)

        self.model = model


class FoldTransformer(Transformer, _FoldBase):

    def __init__(self, dataset_name, hyper_param, exp_dir, fold):
        _FoldBase.__init__(self, dataset_name, hyper_param, exp_dir, fold)
