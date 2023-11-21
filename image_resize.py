#Google Colaboratory
from PIL import Image
import os
import cv2
import numpy as np

#リサイズ後の画像サイズ
resize_vertical = 256
resize_horizontal = 256
#フォルダパス
folder_path1 = "/content/drive/MyDrive/リサイズ前のフォルダパス(既に分類先毎に分かれている)"
folder_path2 = "/content/drive/MyDrive/リサイズ後のフォルダパス(既に分類先毎に分かれている)"
for filename in os.listdir(folder_path1):
  if filename.endswith(".png"):
    input_path = os.path.join(folder_path1, filename)
    output_path1 = os.path.join(folder_path2, f"yohaku_{filename}")
    output_path2 = os.path.join(folder_path2, f"resize_{filename}")

    #画像読み込み
    img = Image.open(input_path)

    #画像の余白部分の画素取得
    img_color = cv2.imread(input_path)
    pixelValue = img_color[1, 2]
    R = pixelValue[2]
    G = pixelValue[1]
    B = pixelValue[0]

    #画像の上下に余白を追加
    width, height = img.size
    if width == height:
      result = img
      img.save(output_path1)
    elif width > height:
      result = Image.new("RGB", (width, width), (R, G, B))
      result.paste(img, (0, (width - height) // 2))
      result.save(output_path1)
    else:
      result = Image.new("RGB", (height, height), (R, G, B))
      result.paste(img, ((height - width) // 2, 0))
      result.save(output_path1)

    #画像リサイズ
    result_resize = result.resize((resize_vertical, resize_horizontal))
    result_resize.save(output_path2)