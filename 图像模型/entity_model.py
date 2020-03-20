import tensorflow as tf
import numpy as np
from img_preprocess import *
model = tf.keras.models.load_model('./IMG_Pre.h5')

name2table=['乌鸦', '乡村', '亭子', '兔', '兰花',
            '夕阳', '大漠', '晴天', '杜鹃', '杨柳',
            '桃花', '梅花', '梧桐', '流水', '浮萍',
            '燕子', '牛', '牡丹', '犬', '猪',
            '猿猴', '玉门关', '白云', '竹', '红豆',
            '芭蕉', '菊花', '落叶', '虎', '蛙',
            '蜡烛', '蝴蝶', '酒', '马', '鱼',
            '鸭', '鸳鸯', '鸿雁',
            ]  # total 38

path='G:/dataset/picture_test/犬与落叶.jpg'
x = img_preprocess(path)
prediction = model.predict(x)
# result = name2table[np.argmax(prediction)]
result_index = np.argsort(prediction[0])[::-1][0:3]  # 输出前三个结果
result = [name2table[result_index[0]],
          name2table[result_index[1]],
          name2table[result_index[2]],
          ]
print(result)
