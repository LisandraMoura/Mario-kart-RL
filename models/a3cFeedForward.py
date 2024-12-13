# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import distutils.version

use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME"):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        fan_in = np.prod(filter_shape[:3])
        fan_out = np.prod(filter_shape[:2]) * num_filters
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, initializer=tf.random_uniform_initializer(-w_bound, w_bound))
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0))
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)

class FeedForwardPolicy(object):
    """
    Politica feed-forward A3C:
    - Aplica algumas camadas convolucionais na entrada
    - Achata e aplica camadas fully connected
    - Produz logits (para a politica) e um valor escalar
    - Sem estado interno (sem LSTM)
    """
    def __init__(self, ob_space, ac_space):
        self.x = tf.placeholder(tf.float32, [None] + list(ob_space), name="Ob")

        # Camadas convolucionais (mesma logica do LSTMPolicy, sem LSTM)
        h = self.x
        for i in range(4):
            # A mesma arquitetura do LSTMPolicy, mas sem LSTM
            h = tf.nn.elu(conv2d(h, 32, "l{}".format(i+1), [3,3], [2,2]))

        # Achata
        h = flatten(h)

        # Camada fully connected para extrair features antes de gerar logits e valor
        size = 256
        h = tf.nn.relu(linear(h, size, "fc1", normalized_columns_initializer(0.01)))

        # Logits das acoes
        self.logits = linear(h, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(h, 1, "value", normalized_columns_initializer(1.0)), [-1])

        self.sample = categorical_sample(self.logits, ac_space)[0, :]

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        # Sem estados internos para LSTM
        return []

    def act(self, ob):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf], {self.x: [ob]})

    def value(self, ob):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob]})[0]
