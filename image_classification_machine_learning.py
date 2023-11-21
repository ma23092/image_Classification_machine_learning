#Google Colaboratory
#%cd "/content/drive/MyDrive/リサイズ前のフォルダパス(既に分類先毎に分かれている)"

import os
import cv2
import numpy as np
import glob as glob
from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
#from keras.utils import np_utils

#from __future__ import print_function, division
import torch
import torch.nn as nn
from torch import optim
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
#import time
#import copy

#フォルダ名をそのままクラス名にするため、フォルダ名を抽出
path = "./images"
folders = os.listdir(path)
classes = [f for f in folders if os.path.isdir(os.path.join(path, f))]
n_classes = len(classes)
classes

X = []
Y = []

for label, class_name in enumerate(classes):
  #読み込むファイルパス
  files = glob.glob("./images/" + class_name + "/*.png")
  for file in files:
    #ファイル読み込み
    img = cv2.imread(file)
    X.append(img)
    Y.append(label)

X = np.array(X)
X = X.astype("float32")
#画素を0-255→0-1に
X /= 255.0

#ラベルをone-hotに変換
#Y = np.array(Y)
Y = to_categorical(Y, n_classes)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
#X_train = torch.Tensor(X_train)
#X_test = torch.Tensor(X_test)
#y_train = torch.Tensor(y_train)
#y_test = torch.Tensor(y_test)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from tensorflow import keras
from keras.layers import Activation, Dense, Flatten
from keras.models import Sequential

model = Sequential([
    Flatten(input_shape=(256, 256, 3)),
    Dense(10000),
    Activation('relu'),
    Dense(5000),
    Activation('relu'),
    Dense(2000),
    Activation('relu'),
    Dense(1000),
    Activation('relu'),
    Dense(500),
    Activation('relu'),
    Dense(200),
    Activation('relu'),
    Dense(100),
    Activation('relu'),
    Dense(5),
    Activation('softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, epochs=100, batch_size=32)
score = model.evaluate(X_test, y_test, batch_size=32)

#サンプルファイル読み込み
img = cv2.imread('sample.png')
img = img.astype('float32')
img /= 255.0
img = img[None, ...]
result = model.predict(img)

#それぞれのクラスに分類される確率を出力
np.set_printoptions(precision=3, suppress=True)
result * 100

#確率が一番高いインデックス
pred = result.argmax()
pred

#確率が一番高いインデックスのクラス名
classes[pred]