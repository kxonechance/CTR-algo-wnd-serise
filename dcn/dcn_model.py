#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
@author: kx
@time: 2020/09/30
@file: dcn_model.py
@function: the implement of deep-cross-net model
@modify:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def model_fn(features, labels, mode, params):
    """Build Deep Cross Net model with binary_classification_head."""

    num_dense_features = params['num_dense_features']
    num_sparse_features = params['num_sparse_features']
    num_dense_fields = params['num_dense_fields']
    num_sparse_fields = params['num_sparse_fields']
    num_factors = params['num_factors']

    learning_rate = params.get('learning_rate', 0.001)

    deep_layers = params.get('deep_layers', [100, 100])
    num_cross_layers = params.get('num_cross_layers', 3)

    dense_indices = tf.reshape(features['dense_indices'], shape=[-1, num_dense_fields])
    dense_values = tf.reshape(features['dense_values'], shape=[-1, num_dense_fields])

    sparse_indices = tf.reshape(features['sparse_indices'], shape=[-1, num_sparse_fields])
    sparse_values = tf.reshape(features['sparse_values'], shape=[-1, num_sparse_fields])

    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = tf.reshape(labels, shape=[-1, 1])

    with tf.variable_scope('emb_part'):
        # num_sparse_features * num_factors
        feat_emb = tf.get_variable('sparse_emb', [num_sparse_features, num_factors], initializer=tf.initializers.glorot_normal())
        # batch * num_sparse_fields * num_factors
        embeddings = tf.nn.embedding_lookup(feat_emb, sparse_indices)
        # batch * (num_sparse_fields * num_factors)
        embeddings = tf.reshape(embeddings, shape=[-1, num_sparse_fields * num_factors])
        # batch * num_dense_fields
        dense_part = tf.reshape(dense_values, shape=[-1, num_dense_fields])
        # batch * (num_sparse_fields * num_factors + num_dense_fields)
        emb_part = tf.concat([embeddings, dense_part], axis=1)

    with tf.variable_scope('deep_part'):
        for i in range(len(deep_layers)):
            if i == 0:
                deep_part = tf.layers.dense(emb_part, deep_layers[i], activation=tf.nn.relu)
            else:
                deep_part = tf.layers.dense(deep_part, deep_layers[i], activation=tf.nn.relu)

    with tf.variable_scope('cross_part'):
        input_dim = emb_part.get_shape().as_list()[1]
        cross_layers = [emb_part]
        for i in range(1, num_cross_layers+1):
            # d * 1
            w = tf.get_variable('cross_{}'.format(i),
                                shape=[input_dim],
                                initializer=tf.initializers.glorot_normal())
            # d * 1
            bias = tf.get_variable('bias_{}'.format(i),
                                   shape=[input_dim],
                                   initializer=tf.initializers.glorot_normal())

            # batch * d * d = (batch * d * 1) x (batch * d * 1)
            x0_x = tf.matmul(tf.expand_dims(cross_layers[0], -1), tf.expand_dims(cross_layers[i-1], -1), transpose_b=True)

            xi = tf.tensordot(x0_x, w, 1) + bias + cross_layers[i-1]

            cross_layers.append(xi)

    # batch * 1
    logits = tf.layers.dense(tf.concat([deep_part, cross_layers[-1]], axis=1), 1, activation=None)

    my_head = tf.contrib.estimator.binary_classification_head()
    return my_head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
        logits=logits
    )












if __name__ == "__main__":
    pass