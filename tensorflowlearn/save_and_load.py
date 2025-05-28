#%% md
# 可以在训练期间和之后保存模型进度。这意味着模型可以从停止的地方恢复，避免长时间的训练。此外，保存还意味着您可以分享您的模型，其他人可以重现您的工作。在发布研究模型和技术时，大多数机器学习从业者会分享：
# 
# - 用于创建模型的代码
# - 模型的训练权重或形参
# 
# 共享数据有助于其他人了解模型的工作原理，并使用新数据自行尝试。
# 
# 小心：TensorFlow 模型是代码，对于不受信任的代码，一定要小心。请参阅 [安全使用 TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md) 以了解详情。
# 
# ### 选项
# 
# 根据您使用的 API，可以通过不同的方式保存 TensorFlow 模型。本指南使用 [tf.keras](https://tensorflow.google.cn/guide/keras) – 一种用于在 TensorFlow 中构建和训练模型的高级 API。建议使用本教程中使用的新的高级 `.keras` 格式来保存 Keras 对象，因为它提供了强大、高效的基于名称的保存，通常比低级或旧版格式更容易调试。如需更高级的保存或序列化工作流，尤其是那些涉及自定义对象的工作流，请参阅[保存和加载 Keras 模型指南](https://tensorflow.google.cn/guide/keras/save_and_serialize)。对于其他方式，请参阅[使用 SavedModel 格式指南](../../guide/saved_model.ipynb)。
import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)
#%% md
# ### 获取示例数据集
# 
# 为了演示如何保存和加载权重，您将使用 [MNIST 数据集](http://yann.lecun.com/exdb/mnist/)。为了加快运行速度，请使用前 1000 个样本：
#%%
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
#%% md
# ### 定义模型
#%% md
# 首先构建一个简单的序列（sequential）模型：
#%%
# Define a simple sequential model
def create_model():
  model = tf.keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()
#%% md
# ## 在训练期间保存模型（以 checkpoints 形式保存）
#%% md
# 您可以使用经过训练的模型而无需重新训练，或者在训练过程中断的情况下从离开处继续训练。`tf.keras.callbacks.ModelCheckpoint` 回调允许您在训练*期间*和*结束*时持续保存模型。
# 
# ### Checkpoint 回调用法
# 
# 创建一个只在训练期间保存权重的 `tf.keras.callbacks.ModelCheckpoint` 回调：
#%%
checkpoint_path = "training_1/cp.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.
#%% md
# 这将创建一个 TensorFlow checkpoint 文件集合，这些文件在每个 epoch 结束时更新：
#%%
os.listdir(checkpoint_dir)
#%% md
# 只要两个模型共享相同的架构，您就可以在它们之间共享权重。因此，当从仅权重恢复模型时，创建一个与原始模型具有相同架构的模型，然后设置其权重。
# 
# 现在，重新构建一个未经训练的全新模型并基于测试集对其进行评估。未经训练的模型将以机会水平执行（约 10% 的准确率）：
#%%
# Create a basic model instance
model = create_model()

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
#%% md
# 然后从 checkpoint 加载权重并重新评估：
#%%
# Loads the weights
model.load_weights(checkpoint_path)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
#%% md
# ### checkpoint 回调选项
# 
# 回调提供了几个选项，为 checkpoint 提供唯一名称并调整 checkpoint 频率。
# 
# 训练一个新模型，每五个 epochs 保存一次唯一命名的 checkpoint ：
#%%
# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

# Calculate the number of batches per epoch
import math
n_batches = len(train_images) / batch_size
n_batches = math.ceil(n_batches)    # round up the number of batches to the nearest whole integer

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*n_batches)

# Create a new model instance
model = create_model()

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the new callback
model.fit(train_images, 
          train_labels,
          epochs=50, 
          batch_size=batch_size, 
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)
#%% md
# 现在，检查生成的检查点并选择最新检查点：
#%%
os.listdir(checkpoint_dir)
#%%
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest
#%% md
# 注：默认 TensorFlow 格式只保存最近的 5 个检查点。
# 
# 要进行测试，请重置模型并加载最新检查点：
#%%
# Create a new model instance
model = create_model()

# Load the previously saved weights
print(checkpoint_dir, latest)
model.load_weights('training_2/cp-0050.weights.h5')

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
#%% md
# ## 这些文件是什么？
#%% md
# 上述代码可将权重存储到[检查点](../../guide/checkpoint.ipynb)格式文件（仅包含二进制格式训练权重） 的合集中。检查点包含：
# 
# - 一个或多个包含模型权重的分片。
# - 一个索引文件，指示哪些权重存储在哪个分片中。
# 
# 如果您在一台计算机上训练模型，您将获得一个具有如下后缀的分片：`.data-00000-of-00001`
#%% md
# ## 手动保存权重
# 
# 要手动保存权重，请使用 `tf.keras.Model.save_weights`。默认情况下，`tf.keras`（尤其是 `Model.save_weights` 方法）使用扩展名为 `.ckpt` 的 TensorFlow [检查点](../../guide/checkpoint.ipynb)格式。要以扩展名为 `.h5` 的 HDF5 格式保存，请参阅[保存和加载模型](https://tensorflow.google.cn/guide/keras/save_and_serialize)指南。
#%%
# Save the weights
model.save_weights('./checkpoints/my_checkpoint.weights.h5')

# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights('./checkpoints/my_checkpoint.weights.h5')

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
#%% md
# ## 保存整个模型
# 
# 调用 `tf.keras.Model.save`，将模型的架构、权重和训练配置保存在单个 `model.keras` zip 存档中。
# 
# 整个模型可以保存为三种不同的文件格式（新的 `.keras` 格式和两种旧格式：`SavedModel` 和 `HDF5`）。将模型保存为 `path/to/model.keras` 会自动以最新格式保存。
# 
# **注意**：对于 Keras 对象，建议使用新的高级 `.keras` 格式进行更丰富的基于名称的保存和重新加载，这样更易于调试。现有代码继续支持低级 SavedModel 格式和旧版 H5 格式。
# 
# 您可以通过以下方式切换到 SavedModel 格式：
# 
# - 将 `save_format='tf'` 传递到 `save()`
# - 传递不带扩展名的文件名
# 
# 您可以通过以下方式切换到 H5 格式：
# 
# - 将 `save_format='h5'` 传递到 `save()`
# - 传递以 `.h5` 结尾的文件名
# 
# Saving a fully-functional model is very useful—you can load them in TensorFlow.js ([Saved Model](https://tensorflow.google.cn/js/tutorials/conversion/import_saved_model), [HDF5](https://tensorflow.google.cn/js/tutorials/conversion/import_keras)) and then train and run them in web browsers, or convert them to run on mobile devices using TensorFlow Lite ([Saved Model](https://tensorflow.google.cn/lite/models/convert/#convert_a_savedmodel_recommended_), [HDF5](https://tensorflow.google.cn/lite/models/convert/#convert_a_keras_model_))
# 
# *Custom objects (for example, subclassed models or layers) require special attention when saving and loading. Refer to the **Saving custom objects** section below.
#%% md
# ### 新的高级 `.keras` 格式
#%% md
# 以 `.keras` 扩展名标记的新 Keras v3 保存格式是一种更简单、更高效的格式，它实现了基于名称的保存，从 Python 的角度确保您加载的内容与您保存的内容完全相同。这使得调试更容易，并且它是 Keras 的推荐格式。
# 
# 下面的部分说明了如何以 `.keras` 格式保存和恢复模型。
#%%
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a `.keras` zip archive.
model.save('my_model.keras')
#%% md
# 从 `.keras` zip 归档重新加载新的 Keras 模型：
#%%
new_model = tf.keras.models.load_model('my_model.keras')

# Show the model architecture
new_model.summary()
#%% md
# 尝试使用加载的模型运行评估和预测：
#%%
# Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(test_images).shape)
#%% md
# ### SavedModel 格式
#%% md
# SavedModel 格式是另一种序列化模型的方式。以这种格式保存的模型可以使用 `tf.keras.models.load_model` 还原，并且与 TensorFlow Serving 兼容。[SavedModel 指南](../../guide/saved_model.ipynb)详细介绍了如何 `serve/inspect` SavedModel。以下部分说明了保存和恢复模型的步骤。
#%%
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
model.save('saved_model/my_model.keras')
#%% md
# SavedModel 格式是一个包含 protobuf 二进制文件和 TensorFlow 检查点的目录。检查保存的模型目录：

#%% md
# 从保存的模型重新加载一个新的 Keras 模型：
#%%
new_model = tf.keras.models.load_model('saved_model/my_model.keras')

# Check its architecture
new_model.summary()
#%% md
# 使用与原始模型相同的实参编译恢复的模型。尝试使用加载的模型运行评估和预测：
#%%
# Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(test_images).shape)
#%% md
# ### HDF5 格式
# 
# Keras 使用 [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) 标准提供基本的旧版高级保存格式。 
#%%
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save('my_model.keras')
#%% md
# 现在，从该文件重新创建模型：
#%%
# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_model.keras')

# Show the model architecture
new_model.summary()
#%% md
# 检查其准确率（accuracy）：
#%%
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
#%% md
# Keras 通过检查模型的架构来保存这些模型。这种技术可以保存所有内容：
# 
# - 权重值
# - 模型的架构
# - 模型的训练配置（您传递给 `.compile()` 方法的内容）
# - 优化器及其状态（如果有）（这样，您便可从中断的地方重新启动训练）
# 
# Keras 无法保存 `v1.x` 优化器（来自 `tf.compat.v1.train`），因为它们与检查点不兼容。对于 v1.x 优化器，您需要在加载-失去优化器的状态后，重新编译模型。
# 
#%% md
# ### 保存自定义对象
# 
# 如果您使用的是 SavedModel 格式，则可以跳过此部分。高级 `.keras`/HDF5 格式与低级 SavedModel 格式之间的主要区别在于 `.keras`/HDF5 格式使用对象配置来保存模型架构，而 SavedModel 保存执行计算图。因此，SavedModels 能够保存自定义对象，例如子类化模型和自定义层，而无需原始代码。但是，因此调试低级 SavedModels 可能会更加困难，鉴于基于名称并且对于 Keras 是原生的特性，我们建议改用高级 `.keras` 格式。
# 
# 要将自定义对象保存到 `.keras` 和 HDF5，您必须执行以下操作：
# 
# 1. 在您的对象中定义一个 `get_config` 方法，并且可以选择定义一个 `from_config` 类方法。
#     - `get_config(self)` 返回重新创建对象所需的形参的 JSON 可序列化字典。
#     - `from_config(cls, config)` 使用从 `get_config` 返回的配置来创建一个新对象。默认情况下，此函数将使用配置作为初始化 kwarg (`return cls(**config)`)。
# 2. 通过以下三种方式之一将自定义对象传递给模型：
#     - 使用 `@tf.keras.utils.register_keras_serializable` 装饰器注册自定义对象。**（推荐）**
#     - 加载模型时直接将对象传递给 `custom_objects` 实参。实参必须是将字符串类名映射到 Python 类的字典。例如 `tf.keras.models.load_model(path, custom_objects={'CustomLayer': CustomLayer})`
#     - 将 `tf.keras.utils.custom_object_scope` 与 `custom_objects` 字典实参中包含的对象一起使用，并在作用域内放置一个 `tf.keras.models.load_model(path){ /code2} 调用。`
# 
# 有关自定义对象和 `get_config` 的示例，请参阅[从头开始编写层和模型](https://tensorflow.google.cn/guide/keras/custom_layers_and_models)教程。
# 