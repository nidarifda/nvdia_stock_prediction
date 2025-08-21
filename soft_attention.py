import tensorflow as tf
from tensorflow.keras.layers import Dense

class SoftAttention(tf.keras.layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.proj = Dense(units, activation="tanh")
        self.score = Dense(1, activation=None)
    def call(self, h, mask=None, training=None):
        e = self.score(self.proj(h))
        if mask is not None:
            m = tf.cast(mask[:, :, tf.newaxis], tf.float32)
            e = e + (1.0 - m) * (-1e9)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(a * h, axis=1)
