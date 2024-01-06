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
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten
from keras.models import Sequential

model = Sequential()
#畳み込み層
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)))  # 256*256*3のRGB画像を想定
# プーリング層
model.add(MaxPooling2D(pool_size=(2, 2)))
# 畳み込み層
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# プーリング層
model.add(MaxPooling2D(pool_size=(2, 2)))
# 畳み込み層
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# プーリング層
model.add(MaxPooling2D(pool_size=(2, 2)))
# 畳み込み層
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
# プーリング層
model.add(MaxPooling2D(pool_size=(2, 2)))
# Flatten層
model.add(Flatten())
# 全結合層
model.add(Dense(128, activation='relu'))
#出力層
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, batch_size=32)

# 履歴の取得
loss_history = history.history['loss']
accuracy_history = history.history['accuracy']
val_loss_history = history.history['val_loss']
val_accuracy_history = history.history['val_accuracy']

import matplotlib.pyplot as plt

# 損失のグラフ表示
plt.plot(loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 精度のグラフ表示
plt.plot(accuracy_history, label='Training Accuracy')
plt.plot(val_accuracy_history, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#サンプルファイル読み込み
sample_img1 = cv2.imread('sample1.png')
sample_img2 = cv2.imread('sample2.png')
sample_img3 = cv2.imread('sample3.png')
sample_img4 = cv2.imread('sample4.png')
sample_img5 = cv2.imread('sample5.png')

sample_img1 = sample_img1.astype('float32')
sample_img2 = sample_img2.astype('float32')
sample_img3 = sample_img3.astype('float32')
sample_img4 = sample_img4.astype('float32')
sample_img5 = sample_img5.astype('float32')

sample_img1 /= 255.0
sample_img2 /= 255.0
sample_img3 /= 255.0
sample_img4 /= 255.0
sample_img5 /= 255.0

sample_img1 = sample_img1[None, ...]
sample_img2 = sample_img2[None, ...]
sample_img3 = sample_img3[None, ...]
sample_img4 = sample_img4[None, ...]
sample_img5 = sample_img5[None, ...]

result1 = model.predict(sample_img1)
result2 = model.predict(sample_img2)
result3 = model.predict(sample_img3)
result4 = model.predict(sample_img4)
result5 = model.predict(sample_img5)

#それぞれのクラスに分類される確率を出力
np.set_printoptions(precision=3, suppress=True)

from PIL import Image
from IPython.display import display

#確率が一番高いインデックスとクラス名
# 画像を読み込む
img1 = Image.open("sample1.png")

# 画像を表示
display(img1)

#print(result1 * 100)
pred1 = result1.argmax()
print(classes[pred1])

#確率が一番高いインデックスとクラス名
# 画像を読み込む
img2 = Image.open("sample2.png")

# 画像を表示
display(img2)

#print(result2 * 100)
pred2 = result2.argmax()
print(classes[pred2])

#確率が一番高いインデックスとクラス名
# 画像を読み込む
img3 = Image.open("sample3.png")

# 画像を表示
display(img3)

#print(result3 * 100)
pred3 = result3.argmax()
print(classes[pred3])

#確率が一番高いインデックスとクラス名
# 画像を読み込む
img4 = Image.open("sample4.png")

# 画像を表示
display(img4)

#print(result4 * 100)
pred4 = result4.argmax()
print(classes[pred4])

#確率が一番高いインデックスとクラス名
# 画像を読み込む
img5 = Image.open("sample5.png")

# 画像を表示
display(img5)

#print(result5 * 100)
pred5 = result5.argmax()
print(classes[pred5])
