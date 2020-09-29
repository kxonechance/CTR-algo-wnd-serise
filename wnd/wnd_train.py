#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
@author: kx
@time: 2020/09/27
@file: wnd_train.py
@function:
@modify:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from wnd import wnd_model
from utils.parse_input import parse_input


tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)


def train(batch_size, epochs, learning_rate, num_factors):

    ret = parse_input('../datasets/train.csv', '../datasets/test.csv', use_cross=True)
    labels = ret['train'][0]
    indices = ret['train'][1]
    values = ret['train'][2]
    num_fields = ret['num_fields']
    num_features = ret['num_features']

    def input_fn(batch_size, epochs, is_training):
        ds = tf.data.Dataset.from_tensor_slices(({'indices': indices, 'values': values}, labels))
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
        model_fn=wnd_model.model_fn,
        model_dir='./model',
        params={
            'num_features': num_features,
            'num_fields': num_fields,
            'num_factors': num_factors,
            'learning_rate': learning_rate
        }
    )

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=10)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    train(batch_size=1, epochs=100, learning_rate=0.001, num_factors=16)
