from model._base import _Base, _FoldBase
from tensorflow.keras.layers import Input
from layer.blocks import Embedding, Encoder, Decoder, Classifier
from config.model_config import implements_hp, heads_hp, emb_dims_hp
from config.data_config import data_config
from config.exec_config import train_config, strategy
from data.loader import seq_loader
from tensorflow.python.keras.layers import Dense
from tensorflow.keras import regularizers
from keras import Model
from datetime import datetime

window, ws = data_config['window'], data_config['ws']
metrics, epoch = train_config['metrics'], train_config['epoch']


class Transformer(_Base):

    def __init__(self, dataset_name, hyper_param, exp_dir):
        super().__init__(dataset_name, hyper_param, exp_dir)
        self.implement = self.hyper_param[implements_hp]
        with strategy.scope():
            self._build()
            self._compile()
            self._load_call()

    def _build(self):
        (w, s) = ws[self.dataset_name]
        seq_len = (window[self.dataset_name] - w) // s + 1

        # blocks
        self.embedding1 = Embedding(
            seq_len,
            self.hyper_param[emb_dims_hp],
            name='embedding1'
        )

        self.embedding2 = Embedding(
            seq_len,
            self.hyper_param[emb_dims_hp],
            name='embedding2'
        )

        self.encoder = Encoder(
            self.hyper_param[heads_hp],
            self.hyper_param[emb_dims_hp],
            self.hyper_param[emb_dims_hp] * 2,
            name='encoder'
        )

        self.decoder = Decoder(
            self.hyper_param[heads_hp],
            self.hyper_param[emb_dims_hp],
            self.hyper_param[emb_dims_hp] * 2,
            name='decoder'
        )

        self.dnn = Dense(
            w * 2, kernel_regularizer=regularizers.l2(0.1),
            name='dense'
        )

        self.classifier = Classifier(
            self.categories,
            self.hyper_param[emb_dims_hp] * 2
        )

        # models
        inputs = Input(shape=(seq_len, w * 2))
        embeddings1 = self.embedding1(inputs)
        embeddings2 = self.embedding2(inputs)
        enc_outputs = self.encoder(embeddings1)
        dec_outputs = self.decoder([embeddings2, enc_outputs])

        if self.implement == 0:
            outputs = self.classifier(inputs)
            self.model = Model(inputs=inputs, outputs=outputs)

        if self.implement in [1, 2]:
            dec_outputs = self.dnn(dec_outputs)
            outputs = self.classifier(enc_outputs)
            self.seq2seq = Model(inputs=inputs, outputs=dec_outputs)
            self.model = Model(inputs=inputs, outputs=outputs)

        if self.implement == 3:
            outputs = self.classifier(enc_outputs)
            self.model = Model(inputs=inputs, outputs=outputs)

        if self.implement == 4:
            outputs = self.classifier(dec_outputs)
            self.model = Model(inputs=inputs, outputs=outputs)

    def _compile(self):
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=metrics
        )

    def _compile_seq(self):
        self.seq2seq.compile(
            loss='mse',
            optimizer='adam'
        )

    def run(self):
        seq_dataset = seq_loader(self.dataset_name, 'train')

        if self.implement in [0, 3, 4]:
            print(f'{datetime.now()} Fitting Model')
            self._compile()
            self.model.fit(
                x=self.dataset_train, validation_data=self.dataset_valid, epochs=3*epoch,
                verbose=0, max_queue_size=10, workers=5, callbacks=self.callbacks
            )

        elif self.implement == 1:
            print(f'{datetime.now()} Fitting Sequence')
            self._compile_seq()
            self.seq2seq.fit(x=seq_dataset, validation_data=seq_dataset, validation_freq=80, epochs=12*epoch, verbose=0)
            self.seq2seq.trainable = False

            print(f'{datetime.now()} Fitting Model')
            self._compile()
            self.model.fit(
                x=self.dataset_train, validation_data=self.dataset_valid, epochs=5*epoch,
                verbose=0, max_queue_size=10, workers=5, callbacks=self.callbacks
            )

        elif self.implement == 2:
            self._compile_seq()
            self._compile()
            for e in range(epoch):
                self.seq2seq.fit(x=seq_dataset, initial_epoch=3*e, epochs=3*(e+1), verbose=0)
                self.model.fit(
                    x=self.dataset_train, validation_data=self.dataset_valid, initial_epoch=3*e, epochs=3*(e+1),
                    verbose=0, max_queue_size=10, workers=5, callbacks=self.callbacks
                )


class FoldTransformer(Transformer, _FoldBase):

    def __init__(self, dataset_name, hyper_param, exp_dir, fold):
        _FoldBase.__init__(self, dataset_name, hyper_param, exp_dir, fold)
