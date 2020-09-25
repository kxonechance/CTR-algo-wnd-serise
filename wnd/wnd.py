#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
@author: kx
@time: 2020/09/25
@file: wnd.py
@function: the implement of wide and deep model
@modify:
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def model_fn(features, labels, mode, params):
    """Build Wide and Deep model"""

    num_features = params['features']
    num_fields = params['num_fields']
    num_factors = params['num_factors']
    num_splits = params['num_splits']

    learning_rate = params.get('learning_rate', 0.001)

    deep_layers = params.get('deep_layers', [100, 100])

    indices = tf.reshape(features['indices'], shape=[-1, num_fields])
    values = tf.reshape(features['values'], shape=[-1, num_fields])

    with tf.variable_scope('linear_part', partitioner=tf.min_max_variable_partitioner(num_splits)):
        # linear_part weight: num_features * 1
        w = tf.get_variable('w', [num_features, 1], initializer=tf.initializers.glorot_normal())
        # batch * num_fields * 1
        linear_part = tf.nn.embedding_lookup(w, indices)
        # batch * num_fields * 1
        linear_part = tf.multiply(linear_part, values)
        # batch * 1
        linear_part = tf.reduce_sum(linear_part, axis=1)

    with tf.variable_scope('deep_part', partitioner=tf.min_max_variable_partitioner(num_splits)):
        # num_features * num_factors
        feat_emb = tf.get_variable('emb', [num_features, num_factors], initializer=tf.initializers.glorot_normal())
        # batch * num_fields * num_factors
        embeddings = tf.nn.embedding_lookup(feat_emb, indices)
        # batch * (num_fields * num_factors)
        deep_part = tf.reshape(embeddings, shape=[-1, num_fields * num_factors])
        for i in range(len(deep_layers)):
            deep_part = tf.layers.dense(deep_part, deep_layers[i], activation=tf.nn.relu)
            deep_part = tf.layers.dropout(deep_part, rate=0.3, training=mode == tf.estimator.Modekeys.TRAIN)

        # batch * 1
        logits = tf.layers.dense(tf.concat([linear_part, deep_part]), 1, activation=None)









if __name__ == "__main__":
    pass