#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
@author: kx
@time: 2020/09/25
@file: wnd_model.py
@function: the implement of wide and deep model
@modify:
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def model_fn(features, labels, mode, params):
    """Build Wide and Deep model with binary_classification_head."""

    num_features = params['num_features']
    num_fields = params['num_fields']
    num_factors = params['num_factors']

    learning_rate = params.get('learning_rate', 0.001)

    deep_layers = params.get('deep_layers', [100, 100])

    indices = tf.reshape(features['indices'], shape=[-1, num_fields])
    values = tf.reshape(features['values'], shape=[-1, num_fields])

    labels = tf.reshape(labels, shape=[-1, 1])

    with tf.variable_scope('linear_part'):
        # linear_part weight: num_features * 1
        w = tf.get_variable('w', [num_features, 1], initializer=tf.initializers.glorot_normal())
        # batch * num_fields * 1
        linear_part = tf.nn.embedding_lookup(w, indices)
        # batch * num_fields * 1
        linear_part = tf.multiply(linear_part, tf.reshape(values, shape=[-1, num_fields, 1]))
        # batch * 1
        linear_part = tf.reduce_sum(linear_part, axis=1)

    with tf.variable_scope('deep_part'):
        # num_features * num_factors
        feat_emb = tf.get_variable('emb', [num_features, num_factors], initializer=tf.initializers.glorot_normal())
        # batch * num_fields * num_factors
        embeddings = tf.nn.embedding_lookup(feat_emb, indices)
        # batch * (num_fields * num_factors)
        deep_part = tf.reshape(embeddings, shape=[-1, num_fields * num_factors])
        for i in range(len(deep_layers)):
            deep_part = tf.layers.dense(deep_part, deep_layers[i], activation=tf.nn.relu)
            deep_part = tf.layers.dropout(deep_part, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

        # batch * 1
        logits = tf.layers.dense(tf.concat([linear_part, deep_part], axis=1), 1, activation=None)

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