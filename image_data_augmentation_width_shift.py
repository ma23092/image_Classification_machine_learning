#Google Colaboratory
from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array, save_img

#アップロードされた画像を読み込み
#フォルダパス
folder_path = "/content/drive/MyDrive/リサイズ前のフォルダパス(既に分類先毎に分かれている)"
folder_path1 = "/content/drive/MyDrive/リサイズ後のフォルダパス(既に分類先毎に分かれている)"
for filename in os.listdir(folder_path):
  if filename.endswith(".png"):
    input_path = os.path.join(folder_path, filename)
    output_path1 = os.path.join(folder_path1, f"4{filename}")
    output_path2 = os.path.join(folder_path1, f"5{filename}")
    output_path3 = os.path.join(folder_path1, f"6{filename}")
    print("ファイル読み込み")
    img = image.load_img(input_path)
    #画像をnumpy配列に変換する
    img = np.array(img)
    #表示画像のサイズを設定
    #plt.figure(figsize = (1, 1))
    #軸を表示しない
    #plt.xticks(color = "None")
    #plt.yticks(color = "None")
    #plt.tick_params(bottom = False, left = False)
    #表示
    #plt.imshow(img)

    print(img.shape)
    #配列に次元を追加
    img_rotate=img[np.newaxis, :, :, :]
    #次元追加後の配列の形
    print(img_rotate.shape)

    #ランダムに左右平行移動するImageDataGeneratorを作成
    width_datagen = ImageDataGenerator(width_shift_range = 0.2)

    for i, data in enumerate(width_datagen.flow(img_rotate, batch_size = 1)):
      #表示のためnumpy配列からimgに変換する
      print("numpy→img")
      show_img = array_to_img(data[0], scale = False)
      #2×3の画像表示の枠を設定＋枠の指定
      #plt.subplot(2, 3, i+1)
      #軸を表示しない       
      #plt.xticks(color = "None")
      #plt.yticks(color = "None")
      #plt.tick_params(bottom = False, left = False)
      #画像を表示
      #plt.imshow(show_img)
      print("次元削除")
      img_width_shift=np.squeeze(show_img)
      #1枚に対し3枚に増やす　結果的に10枚を30枚に
      print(i)
      if i == 0:
        print("画像保存0")
        save_img(output_path1, img_width_shift)
      if i == 1:
        print("画像保存1")
        save_img(output_path2, img_width_shift)
      if i == 2:
        print("画像保存2")
        save_img(output_path3, img_width_shift)
        break