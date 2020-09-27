#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
@author: kx
@time: 2020/09/27
@file: parse_input.py
@function:
@modify:
"""

import pandas as pd

ID_COLUMN = 'id'

TARGET_COLUMN = 'target'

SPARSE_COLUMNS = [
    'ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat',
    'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat',
    'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',
    'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat',
    'ps_car_10_cat', 'ps_car_11_cat',
]

DENSE_COLUMNS = [
    "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",
]

# cross_columns for wnd model
CROSS_COLUMNS = [
    ['ps_ind_04_cat', 'ps_car_04_cat'],
    ['ps_ind_05_cat', 'ps_car_05_cat'],
]


def parse_input(train_path, test_path, use_cross=False):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    y = df_train[TARGET_COLUMN].tolist()
    ids = df_test[ID_COLUMN].tolist()

    df_train = df_train[DENSE_COLUMNS + SPARSE_COLUMNS]
    df_test = df_test[DENSE_COLUMNS + SPARSE_COLUMNS]

    if use_cross:
        for cross_col in CROSS_COLUMNS:
            df_train['-'.join(cross_col)] = df_train[cross_col[0]].astype(str).str.cat(df_train[cross_col[1]].astype(str), sep='-')
            df_test['-'.join(cross_col)] = df_test[cross_col[0]].astype(str).str.cat(df_test[cross_col[1]].astype(str), sep='-')

    df = pd.concat([df_train, df_test])
    idx = 0
    feat2idx = {}
    for col in df.columns:
        if col in DENSE_COLUMNS:
            feat2idx[col] = idx
            idx += 1
        else:
            feats = df[col].unique()
            feat2idx[col] = dict(zip(feats, range(idx, idx+len(feats))))
            idx += len(feats)

    df_train_idx = df_train.copy()
    df_test_idx = df_test.copy()

    for col in df_train.columns:
        if col in DENSE_COLUMNS:
            df_train_idx[col] = feat2idx[col]
        else:
            df_train_idx[col] = df_train_idx[col].apply(lambda x: feat2idx[col][x])
            df_train[col] = 1.0

    for col in df_test.columns:
        if col in DENSE_COLUMNS:
            df_test_idx[col] = feat2idx[col]
        else:
            df_test_idx[col] = df_test_idx[col].apply(lambda x: feat2idx[col][x])
            df_test[col] = 1.0

    df_train_idx = df_train_idx.values.tolist()
    df_train = df_train.values.tolist()

    df_test_idx = df_test_idx.values.tolist()
    df_test = df_test.values.tolist()

    return {'train': (y, df_train_idx, df_train),
            'test': (ids, df_test_idx, df_test),
            'num_features': idx,
            'num_fields': len(df.columns)
            }


if __name__ == "__main__":
    ret = parse_input('../datasets/train.csv', '../datasets/test.csv', use_cross=True)
    print(ret)