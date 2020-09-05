"""
Reference: Apoorv Nandan
https://keras.io/examples/nlp/text_classification_with_transformer/
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
from tensorflow import keras


class PositionalEncoding(Layer):
    def __init__(self, emb_dim):
        super(PositionalEncoding, self).__init__()
        self.emb_dim = emb_dim

    def call(self, inputs, **kwargs):
        batch = tf.shape(inputs)[0]
        encoding = self.positional(batch, self.emb_dim)
        encoding = inputs + encoding

        return encoding

    def positional(self, seq_len, emb_dim):
        rads = self.get_rad(np.arange(seq_len)[:, np.newaxis],
                            np.arange(emb_dim)[np.newaxis, :],
                            emb_dim)

        rads[:, 0::2] = np.sin(rads[:, 0::2])
        rads[:, 1::2] = np.cos(rads[:, 1::2])
        pos_encoding = rads[np.newaxis, :]

        return tf.cast(pos_encoding, dtype=tf.float32)

    @staticmethod
    def get_rad(t, i, emd_dim):
        freq = 1 / np.power(10000, (2 * (i // 2)) / np.float32(emd_dim))
        return t * freq


class TransformerBlock(Layer):
    def __init__(self, head, emb_dim, ffn_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadedAttention(head, emb_dim)
        self.ffn = keras.Sequential(
            [Dense(ffn_dim, activation="relu"), Dense(emb_dim)]
        )
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(rate)
        self.drop2 = Dropout(rate)

    def call(self, inputs, **kwargs):
        att_outputs = self.att(inputs)
        att_outputs = self.drop1(att_outputs)
        att_outputs = self.norm1(inputs + att_outputs)

        ffn_outputs = self.ffn(att_outputs)
        ffn_outputs = self.drop2(ffn_outputs)
        ffn_outputs = self.norm2(att_outputs + ffn_outputs)

        return ffn_outputs


class MultiHeadedAttention(Layer):
    def __init__(self, head, emb_dim):
        super(MultiHeadedAttention, self).__init__()
        if emb_dim % head != 0:
            raise ValueError(f"embedding dimension = {emb_dim} should be divisible by number of head = {head}")
        self.head = head
        self.emb_dim = emb_dim
        self.att_dim = int(emb_dim) // int(head)

        self.que_dense = Dense(emb_dim)
        self.key_dense = Dense(emb_dim)
        self.val_dense = Dense(emb_dim)
        self.con_head = Dense(emb_dim)

    def separate(self, x, batch):
        x = tf.reshape(x, (batch, -1, self.head, self.att_dim))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(self.emb_dim)
        weights = tf.nn.softmax(score, axis=-1)
        outputs = tf.matmul(weights, value)

        return outputs

    def call(self, inputs, **kwargs):
        # que, key, val
        que = self.que_dense(inputs)  # (batch, seq_len, emb_dim)
        key = self.key_dense(inputs)  # (batch, seq_len, emb_dim)
        val = self.val_dense(inputs)  # (batch, seq_len, emb_dim)

        # attention
        batch = tf.shape(inputs)[0]
        que = self.separate(que, batch)  # (batch, head, seq_len, att_dim)
        key = self.separate(key, batch)  # (batch, head, seq_len, att_dim)
        val = self.separate(val, batch)  # (batch , head, seq_len, att_dim)
        outputs = self.attention(que, key, val)

        # concatenate
        outputs = tf.transpose(outputs, perm=[0, 2, 1, 3])  # (batch, seq_len, head, att_dim)
        att_outputs = tf.reshape(outputs, (batch, -1, self.emb_dim))  # (batch, seq_len, emb_dim)
        att_outputs = self.con_head(att_outputs)  # (batch, seq_len, emb_dim)

        return att_outputs


class SoftMax(Layer):
    def __init__(self, categories):
        super(SoftMax, self).__init__()
        self.categories = categories

    def call(self, inputs, **kwargs):
        x = Dropout(0.1)(inputs)
        x = Dense(20, activation="relu")(x)
        x = Dropout(0.1)(x)
        outputs = Dense(units=len(self.categories), activation="softmax", name='softmax')(x)

        return outputs
