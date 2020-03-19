import os
import sys
sys.path.append('..')
from load_img import *
from img_preprocess import *
import tensorflow as tf
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False

tf.random.set_seed(1234)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
assert tf.__version__.startswith('2.')  # 判断使用的是不是tf2.X版本


def img_train_preprocess(x, y):
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)
    x = tf.image.resize(x, [244, 244])

    # data augmentation
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_crop(x, [224, 224, 3])

    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalize(x)
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=3)

    return x, y

root = 'G:/dataset/GAPED_2/emotion2'
batchsz = 64
# training dataset
images, labels, table = load_image(root, 0.8, 1, mode='train')
print(len(labels))
db_train = tf.data.Dataset.from_tensor_slices((images, labels))
db_train = db_train.shuffle(1000).map(img_train_preprocess).batch(batchsz)
# validating dataset
images2, labels2, table = load_image(root, 0.8, 1, mode='val')
print(len(labels2))
db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
db_val = db_val.map(img_train_preprocess).batch(batchsz)
print(db_val)

# Model Building
net = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, pooling='max')
net.trainable = False
newnet = tf.keras.Sequential([
    net,
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(3, activation='softmax'),
])
newnet.build(input_shape=(4, 224, 224, 3))
newnet.summary()
newnet.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
               loss=tf.losses.CategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.001,
    patience=3
)
weights_path = './checkpoints/best_points.h5'
check_point = tf.keras.callbacks.ModelCheckpoint(
    weights_path,
    verbose=1,
    save_best_only=True,
    monitor='val_accuracy',
    save_weights_only=True,
    save_freq=3,
)
if os.path.exists(weights_path):
    newnet.load_weights(weights_path)
    print("checkpoint_loaded")
history = newnet.fit(db_train, validation_data=db_val, epochs=100,
                     callbacks=[check_point, early_stopping])
newnet.save('EMOI_Pre.h5')

history = history.history
print(history.keys())
print(history['val_accuracy'])
print(history['accuracy'])

plt.figure()
returns = history['val_accuracy']
plt.plot(np.arange(len(returns)), returns, label='验证准确率')
plt.plot(np.arange(len(returns)), returns, 's')
returns = history['accuracy']
plt.plot(np.arange(len(returns)), returns, label='训练准确率')
plt.plot(np.arange(len(returns)), returns, 's')

plt.legend()
plt.xlabel('Epoch')
plt.ylabel('准确率')
plt.savefig('figure.svg')