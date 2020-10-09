#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 20/10/09 23:27
@Author  : kx
@Site    : 
@File    : lr_predict.py
@Software: PyCharm
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lr import lr_model
from utils.parse_input import parse_input_v2


tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)


def predict():

    ret = parse_input_v2('../datasets/train.csv', '../datasets/test.csv', use_cross=True)
    ids = ret['test'][0]
    dense_indices = ret['test'][1]
    dense_values = ret['test'][2]
    sparse_indices = ret['test'][3]
    sparse_values = ret['test'][4]
    num_dense_fields = ret['num_fields'][0]
    num_sparse_fields = ret['num_fields'][1]
    num_dense_features = ret['num_features'][0]
    num_sparse_features = ret['num_features'][1]

    def input_fn():
        ds = tf.data.Dataset.from_tensor_slices(({'dense_indices': dense_indices,
                                                  'dense_values': dense_values,
                                                  'sparse_indices': sparse_indices,
                                                  'sparse_values': sparse_values}, ids))
        return ds

    def predict_input_fn():
        return input_fn()

    estimator = tf.estimator.Estimator(
        model_fn=lr_model.model_fn,
        model_dir='./model',
        params={
            'num_dense_features': num_dense_features,
            'num_sparse_features': num_sparse_features,
            'num_dense_fields': num_dense_fields,
            'num_sparse_fields': num_sparse_fields
        }
    )

    preds = estimator.predict(predict_input_fn)

    return list(zip(ids, preds))


if __name__ == "__main__":
    res = predict()
    for i in res:
        print(i)





if __name__ == "__main__":
    pass

