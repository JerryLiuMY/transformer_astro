from model._base import _Base, _FoldBase
from tensorflow.keras.layers import Input
from layer.blocks import Embedding, Encoder, Decoder, Classifier
from config.model_config import implements_hp, heads_hp, emb_dims_hp
from config.data_config import data_config
from config.exec_config import train_config, strategy
from data.loader import seq_loader
from keras import Model

from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback
from tensorboard.plugins.hparams import api as hp
import os
from tools.log_tools import lnr_schedule


window, ws = data_config['window'], data_config['ws']
metrics, epoch = train_config['metrics'], train_config['epoch']


class Transformer(_Base):

    def __init__(self, dataset_name, hyper_param, exp_dir):
        super().__init__(dataset_name, hyper_param, exp_dir)
        self.implement = self.hyper_param[implements_hp]
        with strategy.scope():
            self._build()
            self._compile()

    def _build(self):
        (w, s) = ws[self.dataset_name]
        seq_len = (window[self.dataset_name] - w) // s + 1

        # blocks
        self.embedding1 = Embedding(
            seq_len,
            self.hyper_param[emb_dims_hp]
        )

        self.embedding2 = Embedding(
            seq_len,
            self.hyper_param[emb_dims_hp]
        )

        self.encoder = Encoder(
            self.hyper_param[heads_hp],
            self.hyper_param[emb_dims_hp],
            self.hyper_param[emb_dims_hp] * 2
        )

        self.decoder = Decoder(
            self.hyper_param[heads_hp],
            self.hyper_param[emb_dims_hp],
            self.hyper_param[emb_dims_hp] * 2,
            w * 2
        )

        self.classifier = Classifier(
            self.categories,
            self.hyper_param[emb_dims_hp] * 2
        )

        # model
        inputs = Input(shape=(seq_len, w * 2))
        embeddings1 = self.embedding1(inputs)
        enc_outputs = self.encoder(embeddings1)
        outputs = self.classifier(enc_outputs)
        self.model = Model(inputs=inputs, outputs=outputs)

        if self.implement in [1, 2]:
            embeddings2 = self.embedding2(inputs)
            dec_outputs = self.decoder([embeddings2, enc_outputs])

            self.seq2seq = Model(inputs=inputs, outputs=dec_outputs)

    def _compile(self):
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=self.metrics
        )

    def _compile_seq(self):
        self.seq2seq.compile(
            loss='mse',
            optimizer='adam'
        )

    def run(self):
        seq_dataset = seq_loader(self.dataset_name, 'train')

        for e in range(epoch):
            if self.implement in [1, 2]:
                self.seq2seq.trainable = True
                self._compile_seq()
            if self.implement == 1:
                self.seq2seq.fit(x=seq_dataset, initial_epoch=e, epochs=e+1)
                self.seq2seq.trainable = False
            elif self.implement == 2:
                self.seq2seq.fit(x=seq_dataset, initial_epoch=e, epochs=e+1)
                self.seq2seq.trainable = True

            self._compile()
            self.model.fit(
                x=self.dataset_train, validation_data=self.dataset_valid, initial_epoch=3*e, epochs=3*(e+1),
                verbose=1, max_queue_size=10, workers=5, callbacks=self.callbacks
            )


class FoldTransformer(Transformer, _FoldBase):

    def __init__(self, dataset_name, hyper_param, exp_dir, fold):
        _FoldBase.__init__(self, dataset_name, hyper_param, exp_dir, fold)
