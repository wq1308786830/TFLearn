import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 代码中设置
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 关闭OneDNN优化（日志中建议）
import pandas as pd
import numpy as np
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

# abalone_train = pd.read_csv(
#     "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
#     names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
#            "Viscera weight", "Shell weight", "Age"])
#
# abalone_train.head()
#
# abalone_features = abalone_train.copy()
# abalone_labels = abalone_features.pop('Age')
#
# abalone_features = np.array(abalone_features)
# print(abalone_features)
#
# normalize = layers.Normalization()
# normalize.adapt(abalone_features)
#
# abalone_model = tf.keras.Sequential([
#     normalize,
#     layers.Dense(64, activation='relu'),
#     layers.Dense(1),
# ])
#
# abalone_model.compile(optimizer=tf.keras.optimizers.Adam(),
#                       loss=tf.keras.losses.MeanSquaredError())
#
# abalone_model.fit(abalone_features, abalone_labels, epochs=10)

titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic.head()

titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')

# Create a symbolic input
input = tf.keras.Input(shape=(), dtype=tf.float32)

# Perform a calculation using the input
result = 2*input + 1

# the result doesn't have a value
print(result)

calc = tf.keras.Model(inputs=input, outputs=result)
print(calc(np.array([1])).numpy())
print(calc(np.array([2])).numpy())

inputs = {}

for name, column in titanic_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

print(inputs)

numeric_inputs = {
  name:input for name,input in inputs.items()
                  if input.dtype==tf.float32
}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
print(titanic[numeric_inputs.keys()])
all_numeric_inputs = norm(x)

print(all_numeric_inputs)

preprocessed_inputs = [all_numeric_inputs]

for name, input in inputs.items():
  if input.dtype == tf.float32:
    continue

  lookup = layers.StringLookup(vocabulary=np.unique(titanic_features[name]))
  print(np.unique(titanic_features[name]))
  one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

  x = lookup(input)
  print(x)
  x = one_hot(x)
  print(x)
  preprocessed_inputs.append(x)

print(preprocessed_inputs)
#
# preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
#
# titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)
#
# tf.keras.utils.plot_model(model=titanic_preprocessing, rankdir="LR", dpi=72, show_shapes=True)

# normalize = layers.Normalization()
# normalize.adapt(titanic_features)
#
# model = tf.keras.Sequential([
#     normalize,
#     layers.Dense(64, activation='relu'),
#     layers.Dense(1),
# ])
#
# model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss=tf.keras.losses.MeanSquaredError())
#
# model.fit(titanic_features, titanic_labels, epochs=3)
#



