"""
RGBの３つの画像を読込み行列に変換
３つの画像を合成して元のRGB画像に戻しその画像を保存する
"""

from PIL import Image
import numpy as np

# 3つの画像を読み込む
red_image = Image.open('img/red.jpg')
green_image = Image.open('img/green.jpg')
blue_image = Image.open('img/blue.jpg')

# 画像をNumPy配列に変換
red_array = np.array(red_image)
green_array = np.array(green_image)
blue_array = np.array(blue_image)

# 赤、青、緑のチャネルを合成して元のRGB画像に戻す
rgb_array = np.dstack((red_array, green_array, blue_array))

# テンソルの(500,10)成分の出力
print("ここにテンソルの(500,10)成分の出力する")

# NumPy配列を画像に変換して保存
Image.fromarray(rgb_array).save('img/rgb_image.jpg')