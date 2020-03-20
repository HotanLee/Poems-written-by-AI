# 图像模型部分
## 实体分类(38 class)
- IMG_Pre：已经训练好的模型（未上传）
- entity_train_figure：训练中的准确率
- load_img：用于载入训练集
- img_preprocess：对输入模型中的图片进行预处理
- entity_model_train：模型训练代码
- entity_model：最终模型使用
- emotion：情感分类
## 情感分类（3 class）
- EMOI_Pre：已经训练好的模型（未上传）
- emoi_train_figure：训练中的准确率
- emoi_model_train：模型训练代码
- emoi_model：最终模型使用
## 模型使用方法：
1. 安装tensorflow2.0
2. 打开entity_model.py/emoi_model.py，将path更改为图片地址
3. run
## 模型训练方法：
1. 安装相关库
2. 打开xxx_train_model.py，将root改位训练集地址
3. run
