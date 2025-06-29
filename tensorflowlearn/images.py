import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 关闭OneDNN优化（日志中建议）
# os.environ['TF_GPU_ALLOW_JIT_COMPILATION'] = '1'  # 强制启用JIT

import pathlib
import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

print("TF Version:", tf.__version__)  # 应输出 2.19
print("GPU Available:", tf.config.list_physical_devices('GPU'))
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(physical_devices[0], 'GPU')  # 指定单个GPU[9](@ref)
# print("cuDNN Version:", tf.sysconfig.get_build_info()['cudnn_version'])
# print("GPU支持列表:", tf.sysconfig.get_build_info()["cuda_compute_capabilities"])

img_width = 180
img_height = 180
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# dataset_url = "file:///mnt/d/Users/flyin/PycharmProjects/TFLearn/tensorflowlearn/flower_photos.tgz"
archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)

data_dir = pathlib.Path(archive).with_suffix('')

data_dir = pathlib.Path(data_dir) / 'flower_photos'

image_count = len(list(data_dir.glob('*/*.jpg')))
rose_list = list(data_dir.glob('roses/*'))

# print("Total images:", rose_list)


img = PIL.Image.open(str(rose_list[0]))
plt.imshow(img)
plt.show()
PIL.Image.open(str(rose_list[1]))


train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_width, img_height),
    batch_size=batch_size
)

validate_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_width, img_height),
    batch_size=batch_size
)

class_names = train_ds.class_names

print("Class names:", class_names)


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = validate_ds.cache().prefetch(buffer_size=AUTOTUNE)


model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)


#
#
#
# (train_ds, val_ds, test_ds), metadata = tfds.load(
#     'tf_flowers',
#     split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
#     with_info=True,
#     as_supervised=True,
# )
#
# num_classes = metadata.features['label'].num_classes
# print(111, num_classes)
#
# get_label_name = metadata.features['label'].int2str
#
# image, label = next(iter(train_ds))
# _ = plt.imshow(image)
# _ = plt.title(get_label_name(label))
#
# def configure_for_performance(ds):
#   ds = ds.cache()
#   ds = ds.shuffle(buffer_size=1000)
#   ds = ds.batch(batch_size)
#   ds = ds.prefetch(buffer_size=AUTOTUNE)
#   return ds
#
# train_ds = configure_for_performance(train_ds)
# val_ds = configure_for_performance(val_ds)
# test_ds = configure_for_performance(test_ds)
