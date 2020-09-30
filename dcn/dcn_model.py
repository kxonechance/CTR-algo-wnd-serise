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

    num_features = params['num_features']
    num_fields = params['num_fields']
    num_factors = params['num_factors']







if __name__ == "__main__":
    pass