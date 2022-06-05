#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@filename     :gnss-data.py
@description  :
@author       :hzh
@time         :2022/05/05 21:58:46
'''

# import os
# import pandas as pd
# from numpy import vstack


# def import_from_files():
#     """
#         Read .csv files and store data into an array
#         format: |LOS|NLOS|data...|
#     """
#     rootdir = '../dataset/'
#     output_arr = []
#     first = 1
#     for dirpath, dirnames, filenames in os.walk(rootdir):
#         for file in filenames:
#             filename = os.path.join(dirpath, file)
#             print(filename)
#             output_data = [] 
#             # read data from file
#             df = pd.read_csv(filename, sep=',', header=0)
#             input_data = df.as_matrix()
#             # append to array
#             if first > 0:
#                 first = 0
#                 output_arr = input_data
#             else:
#                 output_arr = vstack((output_arr, input_data))
    
#     return output_arr

# if __name__ == '__main__':

#     # import raw data from folder with dataset
#     print("Importing dataset to numpy array")
#     print("-------------------------------")
#     data = import_from_files()
#     print("-------------------------------")
#     # print dimensions and data
#     print("Number of samples in dataset: %d" % len(data))
#     print("Length of one sample: %d" % len(data[0]))
#     print("-------------------------------")
#     print("Dataset:")
#     print(data)

import functools

import numpy as np
import tensorflow as tf
# import tensorflow_datasets as tfds

TRAIN_DATA_URL = "file:///D:/Proj/pyplot/data/wuce-nlos.csv"
TEST_DATA_URL = "file:///D:/Proj/pyplot/data/wuce-nlos.csv"


train_file_path = tf.keras.utils.get_file("wuce-nlos.csv", TRAIN_DATA_URL)
# test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

np.set_printoptions(precision=3, suppress=True)


LABEL_COLUMN = 'NLOS'
LABELS = [0, 1]

def get_dataset(file_path):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=12, # 为了示例更容易展示，手动设置较小的值
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
  return dataset

raw_train_data = get_dataset(train_file_path)
# raw_test_data = get_dataset(test_file_path)

examples, labels = next(iter(raw_train_data)) # 第一个批次
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)

def process_continuous_data(mean, data):
  # 标准化数据
  data = tf.cast(data, tf.float32) * 1/(2*mean)
  return tf.reshape(data, [-1, 1])

MEANS = {
    'P' : 27602293.000,
    'L' : 144493987.122,
    'SNR' : 43.28
}

numerical_columns = []

for feature in MEANS.keys():
  num_col = tf.feature_column.numeric_column(feature, normalizer_fn=functools.partial(process_continuous_data, MEANS[feature]))
  numerical_columns.append(num_col)

preprocessing_layer = tf.keras.layers.DenseFeatures(numerical_columns)

model = tf.keras.Sequential([
  preprocessing_layer,
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

train_data = raw_train_data.shuffle(500)
# test_data = raw_test_data

model.fit(train_data, epochs=20)

# import functools

# import numpy as np
# import tensorflow as tf
# # import tensorflow_datasets as tfds
# TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
# TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

# train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
# test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

# # 让 numpy 数据更易读。
# np.set_printoptions(precision=3, suppress=True)

# LABEL_COLUMN = 'survived'
# LABELS = [0, 1]

# def get_dataset(file_path):
#   dataset = tf.data.experimental.make_csv_dataset(
#       file_path,
#       batch_size=12, # 为了示例更容易展示，手动设置较小的值
#       label_name=LABEL_COLUMN,
#       na_value="?",
#       num_epochs=1,
#       ignore_errors=True)
#   return dataset

# raw_train_data = get_dataset(train_file_path)
# raw_test_data = get_dataset(test_file_path)

# examples, labels = next(iter(raw_train_data)) # 第一个批次
# print("EXAMPLES: \n", examples, "\n")
# print("LABELS: \n", labels)

# CATEGORIES = {
#     'sex': ['male', 'female'],
#     'class' : ['First', 'Second', 'Third'],
#     'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
#     'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
#     'alone' : ['y', 'n']
# }

# categorical_columns = []
# for feature, vocab in CATEGORIES.items():
#   cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
#         key=feature, vocabulary_list=vocab)
#   categorical_columns.append(tf.feature_column.indicator_column(cat_col))


# def process_continuous_data(mean, data):
#   # 标准化数据
#   data = tf.cast(data, tf.float32) * 1/(2*mean)
#   return tf.reshape(data, [-1, 1])

# MEANS = {
#     'age' : 29.631308,
#     'n_siblings_spouses' : 0.545455,
#     'parch' : 0.379585,
#     'fare' : 34.385399
# }

# numerical_columns = []

# for feature in MEANS.keys():
#   num_col = tf.feature_column.numeric_column(feature, normalizer_fn=functools.partial(process_continuous_data, MEANS[feature]))
#   numerical_columns.append(num_col)

# preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numerical_columns)

# model = tf.keras.Sequential([
#   preprocessing_layer,
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(1, activation='sigmoid'),
# ])

# model.compile(
#     loss='binary_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy'])

# train_data = raw_train_data.shuffle(500)
# test_data = raw_test_data

# model.fit(train_data, epochs=20)