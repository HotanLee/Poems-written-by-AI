import tensorflow as tf
import numpy as np
import sys
sys.path.append('..')
from img_preprocess import *
model = tf.keras.models.load_model('./EMOI_Pre.h5')
name2table = ['消极', '中性', '积极']

path = 'G:/dataset/GAPED_2/test/P002.jpg'
x = img_preprocess(path)
prediction = model.predict(x)
print(prediction)
result = name2table[np.argmax(prediction)]
print(result)