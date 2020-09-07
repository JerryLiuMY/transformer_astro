from model._base import _Base, _FoldBase
from tensorflow.keras.layers import Input
from layer.encoder import Embedding, Encoder, Classifier
from config.model_config import heads_hp, emb_dims_hp, ffn_dims_hp
from config.data_config import data_config
from config.exec_config import train_config
from keras import Model

use_gen, epoch = train_config['use_gen'], train_config['epoch']
window, ws = data_config['window'], data_config['ws']


class Attention(_Base):

    def __init__(self, dataset_name, hyper_param, exp_dir):
        super().__init__(dataset_name, hyper_param, exp_dir)

    def _build(self):
        (w, s) = ws[self.dataset_name]
        seq_len = (window[self.dataset_name] - w) // s + 1

        inputs = Input(shape=(seq_len, w * 2))

        self.embedding = Embedding(
            seq_len,
            self.hyper_param[emb_dims_hp]
        )

        self.encoder = Encoder(
            self.hyper_param[heads_hp],
            self.hyper_param[emb_dims_hp],
            self.hyper_param[ffn_dims_hp]
        )

        self.classifier = Classifier(
            self.categories,
            self.hyper_param[ffn_dims_hp]
        )

        embeddings = self.embedding(inputs)
        enc_outputs = self.encoder([embeddings])
        outputs = self.classifier(enc_outputs)
        self.model = Model(inputs=inputs, outputs=outputs)


class FoldAttention(Attention, _FoldBase):

    def __init__(self, dataset_name, hyper_param, exp_dir, fold):
        _FoldBase.__init__(self, dataset_name, hyper_param, exp_dir, fold)
