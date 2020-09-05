from model._base import _Base, _FoldBase
from tensorflow.keras.layers import GlobalAveragePooling1D, Input
from layer.encoder import PositionalEncoding, TransformerBlock, SoftMax
from keras import Model
from config.model_config import heads_hp, emb_dims_hp, ffn_dims_hp
from config.data_config import data_config
from config.exec_config import train_config

use_gen, ws = train_config['use_gen'], data_config['ws']


class Transformer(_Base):

    def __init__(self, dataset_name, hyper_param, exp_dir):
        super().__init__(dataset_name, hyper_param, exp_dir)

        inputs = Input(shape=(None, ws[self.dataset_name][0] * 2))
        embedding = PositionalEncoding(self.hyper_param[emb_dims_hp])
        transformer = TransformerBlock(self.hyper_param[heads_hp],
                                       self.hyper_param[emb_dims_hp],
                                       self.hyper_param[ffn_dims_hp])
        softmax = SoftMax(self.categories)
        x = embedding(inputs)
        x = transformer(x)
        x = GlobalAveragePooling1D()(x)
        outputs = softmax(x)
        model = Model(inputs=inputs, outputs=outputs)

        self.model = model


class FoldTransformer(Transformer, _FoldBase):

    def __init__(self, dataset_name, hyper_param, exp_dir, fold):
        _FoldBase.__init__(self, dataset_name, hyper_param, exp_dir, fold)
