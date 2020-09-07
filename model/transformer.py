from model._base import _Base, _FoldBase
from tensorflow.keras.layers import Input
from layer.blocks import Embedding, Encoder, Decoder, Classifier
from config.model_config import heads_hp, emb_dims_hp, ffn_dims_hp
from config.data_config import data_config
from config.exec_config import train_config
from keras import Model
import tensorflow as tf

window, ws = data_config['window'], data_config['ws']
use_gen, epoch = train_config['use_gen'], train_config['epoch']
implementation = train_config['implementation']


class Transformer(_Base):

    def __init__(self, dataset_name, hyper_param, exp_dir):
        super().__init__(dataset_name, hyper_param, exp_dir)

    def _build(self):
        (w, s) = ws[self.dataset_name]
        seq_len = (window[self.dataset_name] - w) // s + 1

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
            self.hyper_param[ffn_dims_hp]
        )

        self.decoder = Decoder(
            self.hyper_param[heads_hp],
            self.hyper_param[emb_dims_hp],
            self.hyper_param[ffn_dims_hp]
        )

        self.classifier = Classifier(
            self.categories,
            self.hyper_param[ffn_dims_hp]
        )

        inputs = Input(shape=(seq_len, w * 2))
        print(implementation)
        if implementation == 0:
            embeddings1 = self.embedding1(inputs)
            enc_outputs = self.encoder(embeddings1)
            outputs = self.classifier(enc_outputs)
            self.model = Model(inputs=inputs, outputs=outputs)

        elif implementation in [1, 2]:
            embeddings1 = self.embedding1(inputs)
            embeddings2 = self.embedding2(inputs)
            enc_outputs = self.encoder(embeddings1)
            dec_outputs = self.decoder([embeddings2, enc_outputs])
            outputs = self.classifier(enc_outputs)
            self.model = [Model(inputs=inputs, outputs=dec_outputs), Model(inputs=inputs, outputs=outputs)]

        else:
            raise AssertionError('Invalid implementation')

    def load_sequence(self):
        dataset = None
        for x, y, sample_weight in self.dataset_train:
            if dataset is None:
                dataset = tf.data.Dataset.from_tensor_slices(x)
            else:
                dataset = dataset.concatenate(tf.data.Dataset.from_tensor_slices(x))

        seq_dataset = tf.data.Dataset.zip((dataset, dataset))

        return seq_dataset

    def run(self):
        if implementation == 0:
            super().run()

        elif implementation in [1, 2]:
            transformer, classifier = self.model
            seq_dataset = self.load_sequence()
            for e in range(epoch):
                transformer.trainable = True
                if implementation == 1:
                    transformer.train_on_batch(seq_dataset)
                    transformer.trainable = False
                else:
                    transformer.train_on_batch(seq_dataset)
                    transformer.trainable = True

                classifier.fit(
                    x=self.dataset_train, validation_data=self.dataset_valid, initial_epoch=e, epochs=e+1,
                    verbose=1, max_queue_size=10, workers=5, callbacks=self._callbacks()
                )

        else:
            raise AssertionError('Invalid implementation')


class FoldTransformer(Transformer, _FoldBase):

    def __init__(self, dataset_name, hyper_param, exp_dir, fold):
        _FoldBase.__init__(self, dataset_name, hyper_param, exp_dir, fold)
