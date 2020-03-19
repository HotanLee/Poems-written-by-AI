import tensorflow as tf

img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])


def img_preprocess(path):
    x = tf.io.read_file(path)
    x = tf.image.decode_jpeg(x, channels=3)
    x = tf.image.resize(x, [244, 244])
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalize(x)
    x = tf.reshape(x, [1, 244, 244, 3])
    return x


def normalize(x, mean=img_mean, std=img_std):
    x = (x-mean)/std
    return x