from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers import LayerNormalization, Dense, ReLU
from layer.multihead import MultiHeadAttention, FFN
from tensorflow.keras import regularizers
import tensorflow as tf
import numpy as np


class Embedding(Layer):
    def __init__(self, seq_len, emb_dim, name):
        super(Embedding, self).__init__(name=name)
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.dense = Dense(self.emb_dim, activation='relu')

    def call(self, inputs, **kwargs):
        # broadcast
        word2vecs = self.dense(inputs)
        encodings = self.positional(self.seq_len, self.emb_dim)
        embeddings = tf.math.add(word2vecs, encodings)

        return embeddings

    def positional(self, seq_len, emb_dim):
        rads = self.get_rad(np.arange(seq_len)[:, np.newaxis],
                            np.arange(emb_dim)[np.newaxis, :],
                            emb_dim)

        rads[:, 0::2] = np.sin(rads[:, 0::2])
        rads[:, 1::2] = np.cos(rads[:, 1::2])
        encodings = rads[np.newaxis, :]
        encodings = tf.cast(encodings, dtype=tf.float32)

        return encodings

    @staticmethod
    def get_rad(t, i, emd_dim):
        freq = 1 / np.power(10000, (2 * (i // 2)) / np.float32(emd_dim))

        return t * freq


class Encoder(Layer):
    def __init__(self, head, emb_dim, ffn_dim, name):
        super(Encoder, self).__init__(name=name)
        self.att = MultiHeadAttention(head, emb_dim)
        self.ffn = FFN(emb_dim, ffn_dim)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, **kwargs):
        embeddings = inputs
        att_outputs = self.att([embeddings, embeddings, embeddings])
        att_outputs = self.norm1(inputs + att_outputs)

        enc_outputs = self.ffn(att_outputs)
        enc_outputs = self.norm2(att_outputs + enc_outputs)

        return enc_outputs


class Decoder(Layer):
    def __init__(self, head, emb_dim, ffn_dim, name):
        super(Decoder, self).__init__(name=name)
        self.att1 = MultiHeadAttention(head, emb_dim)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.att2 = MultiHeadAttention(head, emb_dim)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.ffn = FFN(emb_dim, ffn_dim)
        self.norm3 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, **kwargs):
        embeddings, enc_outputs = inputs
        att_outputs1 = self.att1([embeddings, embeddings, embeddings])
        att_outputs1 = self.norm1(att_outputs1 + embeddings)

        att_outputs2 = self.att2([att_outputs1, enc_outputs, enc_outputs])
        att_outputs2 = self.norm2(att_outputs2 + att_outputs1)

        dec_outputs = self.ffn(att_outputs2)
        dec_outputs = self.norm3(att_outputs2 + dec_outputs)

        return dec_outputs


class Classifier(Layer):
    def __init__(self, categories, ffn_dim):
        super(Classifier, self).__init__(name='classifier')
        self.dnn = Dense(ffn_dim, kernel_regularizer=regularizers.l2(0.1))
        self.norm = LayerNormalization(epsilon=1e-6)
        self.relu = ReLU()
        self.sfm = Dense(units=len(categories), activation='softmax')
        self.poo = GlobalAveragePooling1D(data_format='channels_last')

    def call(self, inputs, **kwargs):
        # int_outputs = self.poo(inputs)
        # int_outputs = self.dnn(int_outputs)
        # int_outputs = self.norm(int_outputs)
        # dnn_outputs = self.relu(int_outputs)
        # outputs = self.sfm(dnn_outputs)

        int_outputs = self.dnn(inputs)
        int_outputs = self.norm(int_outputs)
        dnn_outputs = self.relu(int_outputs)
        sfm_outputs = self.sfm(dnn_outputs)
        outputs = self.poo(sfm_outputs)

        return outputs
