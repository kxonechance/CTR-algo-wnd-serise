#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
@author: kx
@time: 2020/10/09
@file: lr_model.py
@function:
@modify:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def model_fn(features, labels, mode, params):
    """Build logistic regression model with binary_classification_head."""

    num_dense_features = params['num_dense_features']
    num_sparse_features = params['num_sparse_features']
    num_dense_fields = params['num_dense_fields']
    num_sparse_fields = params['num_sparse_fields']

    learning_rate = params.get('learning_rate', 0.001)

    dense_indices = tf.reshape(features['dense_indices'], shape=[-1, num_dense_fields])
    dense_values = tf.reshape(features['dense_values'], shape=[-1, num_dense_fields])

    sparse_indices = tf.reshape(features['sparse_indices'], shape=[-1, num_sparse_fields])
    sparse_values = tf.reshape(features['sparse_values'], shape=[-1, num_sparse_fields])

    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = tf.reshape(labels, shape=[-1, 1])

    with tf.variable_scope('linear_part'):
        # num_features * 1
        w = tf.get_variable('w', shape=[num_dense_features + num_sparse_features], initializer=tf.initializers.glorot_normal())
        global_bias = tf.get_variable('bias', shape=[1], initializer=tf.constant_initializer(0.0))
        # batch * num_fields
        indices = tf.concat([dense_indices, sparse_indices], axis=1)
        # batch * num_fields * 1
        linear_part = tf.nn.embedding_lookup(w, indices)
        # batch * num_fields * 1
        linear_part = tf.reshape(linear_part, shape=[-1, num_dense_fields+num_sparse_fields, 1])
        # batch * num_fields
        values = tf.concat([dense_values, sparse_values], axis=1)
        # batch * num_fields * 1
        linear_part = tf.multiply(linear_part, tf.reshape(values, shape=[-1, num_dense_fields+num_sparse_fields, 1]))
        # batch * 1
        linear_part = tf.reduce_sum(linear_part, axis=1)
        # batch * 1
        bias = global_bias * tf.ones_like(linear_part, dtype=tf.float32)
        # batch * 1
        logits = bias + linear_part

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