import os
import glob
import random
import csv


def load_image(root, rate1=0.6, rate2=0.8, mode='train'):
    name2label = {}  # x elements in it
    for name in sorted(os.listdir(os.path.join(root))):  # 返回指定路径下的文件和文件夹
        if not os.path.isdir(os.path.join(root, name)):
            continue
        name2label[name] = len(name2label.keys())  # ep:{positive：0, negative：1}

    # read labels
    images, labels = load_csv(root, 'image.csv', name2label)
    # divede datasets
    if mode == 'train':  # 60%
        images = images[:int(rate1*len(images))]
        labels = labels[:int(rate1*len(labels))]
    elif mode == 'val':  # 20% = 60%-->80%
        images = images[int(rate1 * len(images)):int(rate2 * len(images))]
        labels = labels[int(rate1 * len(labels)):int(rate2 * len(labels))]
    else:  # 20% = 80%-->100%
        images = images[int(rate2 * len(images)):int(len(images))]
        labels = labels[int(rate2 * len(labels)):int(len(labels))]

    return images, labels, name2label


def load_csv(root, filename, name2label):
    # 如果没有csv
    if not os.path.exists(os.path.join(root, filename)):
        images = []
        for name in name2label.keys():
            images += glob.glob(os.path.join(root, name, '*.jpg'))
        print(len(images))
        random.shuffle(images)
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:
                name = img.split(os.sep)[-2]
                label = name2label[name]
                writer.writerow([img, label])
            print('written into csv file:', filename)

    # 如果已经有csv
    images, labels = [], []
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            imag, label = row
            label = int(label)
            images.append(imag)
            labels.append(label)
    return images, labels