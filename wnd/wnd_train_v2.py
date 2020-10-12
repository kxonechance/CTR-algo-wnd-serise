#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 20/10/01 18:35
@Author  : stevenke
@Site    : 
@File    : wnd_train_v2.py
@Software: PyCharm
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from wnd import wnd_model_v2
from utils.parse_input import parse_input_v2


tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)


def train(batch_size, epochs, learning_rate, num_factors, deep_layers):

    ret = parse_input_v2('../datasets/train.csv', '../datasets/test.csv', use_cross=True)
    labels = ret['train'][0]
    dense_indices = ret['train'][1]
    dense_values = ret['train'][2]
    sparse_indices = ret['train'][3]
    sparse_values = ret['train'][4]
    num_dense_fields = ret['num_fields'][0]
    num_sparse_fields = ret['num_fields'][1]
    num_dense_features = ret['num_features'][0]
    num_sparse_features = ret['num_features'][1]

    def input_fn(batch_size, epochs, is_training):
        ds = tf.data.Dataset.from_tensor_slices(({'dense_indices': dense_indices,
                                                  'dense_values': dense_values,
                                                  'sparse_indices': sparse_indices,
                                                  'sparse_values': sparse_values}, labels))
        if is_training:
            ds = ds.shuffle(100).batch(batch_size).repeat(epochs)
        return ds

        # iterator = ds.make_one_shot_iterator()
        # batch_features, batch_labels = iterator.get_next()
        # return tf.reshape(batch_ids,shape=[-1,field_size]), tf.reshape(batch_vals,shape=[-1,field_size]), batch_labels
        # return batch_features, batch_labels

    def train_input_fn():
        return input_fn(batch_size, epochs, is_training=True)

    def eval_input_fn():
        return input_fn(batch_size, epochs, is_training=False)

    estimator = tf.estimator.Estimator(
        model_fn=wnd_model_v2.model_fn,
        model_dir='./model',
        params={
            'num_dense_features': num_dense_features,
            'num_sparse_features': num_sparse_features,
            'num_dense_fields': num_dense_fields,
            'num_sparse_fields': num_sparse_fields,
            'num_factors': num_factors,
            'learning_rate': learning_rate,
            'deep_layers': deep_layers
        }
    )

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=10)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    train(batch_size=1, epochs=100, learning_rate=0.001, num_factors=16, deep_layers=[100, 100])

