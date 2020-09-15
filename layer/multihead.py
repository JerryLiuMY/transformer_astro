from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from tensorflow import keras
import tensorflow as tf


class MultiHeadAttention(Layer):
    def __init__(self, head, emb_dim, name='multihead'):
        super(MultiHeadAttention, self).__init__(name=name)
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
        que_inputs, key_inputs, val_inputs = inputs
        que = self.que_linear(que_inputs)  # (batch, seq_len, emb_dim)
        key = self.key_linear(key_inputs)  # (batch, seq_len, emb_dim)
        val = self.val_linear(val_inputs)  # (batch, seq_len, emb_dim)

        # attention
        batch = tf.shape(que_inputs)[0]
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
        weights = tf.nn.softmax(scores, axis=-1, name='attention_weight')
        outputs = tf.matmul(weights, value)

        return outputs


class FFN(Layer):
    def __init__(self, emb_dim, ffn_dim, name='encoder'):
        super(FFN, self).__init__(name=name)
        self.ffn = keras.Sequential([
            Dense(emb_dim),
            Dense(ffn_dim, activation='relu', kernel_regularizer=regularizers.l2(0.1)),
            Dense(emb_dim)
        ])

    def call(self, inputs, **kwargs):
        ffn_outputs = self.ffn(inputs)

        return ffn_outputs
