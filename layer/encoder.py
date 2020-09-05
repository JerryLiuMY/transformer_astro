from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
from tensorflow.keras import regularizers
from tensorflow import keras
import tensorflow as tf
import numpy as np

# TODO: change to embedding


class Embedding(Layer):
    def __init__(self, seq_len, emb_dim):
        super(Embedding, self).__init__()
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.dense = Dense(self.emb_dim, activation='relu')

    def call(self, inputs, **kwargs):
        # broadcast
        embedding = self.dense(inputs)
        encodings = self.positional(self.seq_len, self.emb_dim)
        encodings = tf.math.add(embedding, encodings)

        return encodings

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


class TransformerBlock(Layer):
    def __init__(self, head, emb_dim, ffn_dim):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadedAttention(head, emb_dim)
        self.ffn = keras.Sequential([
            Dense(emb_dim),
            Dense(ffn_dim, recurrent_regularizer=regularizers.l2(0.1), activation='relu'),
            Dense(emb_dim)
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, **kwargs):
        att_outputs = self.att(inputs)
        att_outputs = self.norm1(inputs + att_outputs)

        ffn_outputs = self.ffn(att_outputs)
        ffn_outputs = self.norm2(att_outputs + ffn_outputs)

        return ffn_outputs


class MultiHeadedAttention(Layer):
    def __init__(self, head, emb_dim):
        super(MultiHeadedAttention, self).__init__()
        if emb_dim % head != 0:
            raise ValueError(f'embedding dimension = {emb_dim} should be divisible by number of head = {head}')
        self.head = head
        self.emb_dim = emb_dim
        self.att_dim = int(emb_dim) // int(head)

        self.que_linear = Dense(emb_dim)
        self.key_linear = Dense(emb_dim)
        self.val_linear = Dense(emb_dim)
        self.con_linear = Dense(emb_dim)

    def call(self, inputs, **kwargs):
        # que, key, val
        que = self.que_linear(inputs)  # (batch, seq_len, emb_dim)
        key = self.key_linear(inputs)  # (batch, seq_len, emb_dim)
        val = self.val_linear(inputs)  # (batch, seq_len, emb_dim)

        # attention
        batch = tf.shape(inputs)[0]
        que = self.separate(que, batch)  # (batch, head, seq_len, att_dim)
        key = self.separate(key, batch)  # (batch, head, seq_len, att_dim)
        val = self.separate(val, batch)  # (batch , head, seq_len, att_dim)
        dot_outputs = self.attention(que, key, val)

        # concatenate + linear
        dot_outputs = tf.transpose(dot_outputs, perm=[0, 2, 1, 3])  # (batch, seq_len, head, att_dim)
        att_outputs = tf.reshape(dot_outputs, (batch, -1, self.emb_dim))  # (batch, seq_len, emb_dim)
        att_outputs = self.con_linear(att_outputs)  # (batch, seq_len, emb_dim)

        return att_outputs

    def separate(self, x, batch):
        x = tf.reshape(x, (batch, -1, self.head, self.att_dim))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def attention(self, query, key, value):
        key_dim = tf.cast(self.emb_dim, tf.float32)
        scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(key_dim)
        weights = tf.nn.softmax(scores, axis=-1)
        outputs = tf.matmul(weights, value)

        return outputs
