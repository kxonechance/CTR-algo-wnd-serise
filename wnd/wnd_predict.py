#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
@author: kx
@time: 2020/09/27
@file: wnd_predict.py
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


def predict(num_factors):

    ret = parse_input('../datasets/train.csv', '../datasets/test.csv', use_cross=True)
    ids = ret['test'][0]
    indices = ret['test'][1]
    values = ret['test'][2]
    num_fields = ret['num_fields']
    num_features = ret['num_features']

    def input_fn():
        ds = tf.data.Dataset.from_tensor_slices(({'indices': indices, 'values': values}, ids))
        return ds

    def predict_input_fn():
        return input_fn()

    estimator = tf.estimator.Estimator(
        model_fn=wnd_model.model_fn,
        model_dir='./model',
        params={
            'num_features': num_features,
            'num_fields': num_fields,
            'num_factors': num_factors
        }
    )

    preds = estimator.predict(predict_input_fn)

    return list(zip(ids, preds))


if __name__ == "__main__":
    res = predict(16)
    for i in res:
        print(i)