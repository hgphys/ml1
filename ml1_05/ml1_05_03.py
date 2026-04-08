"""
画像ファイルを読込み、特異値分解を行う
各特異値に対応する画像を出力した上で
最後に特異値の割合が99.9%になるまで合成した画像を出力する
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 画像の読み込み
img = Image.open("img/rgb_image.jpg").convert('L')  # グレースケールに変換
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

# 画像を行列に変換
A = np.array(img)
print("画像の行列Aの形状:", A.shape)

# 特異値分解
U, s, Vt = np.linalg.svd(A, full_matrices=False)
print("行列Uの形状:", U.shape)
print("特異値sの形状:", s.shape)
print("行列Vtの形状:", Vt.shape)

# 指定した特異値まで足した画像を出力
num_sv = 3
for i in range(num_sv):
    s_ = np.zeros_like(s)
    s_[:i+1] = s[:i+1]
    A_ = U @ np.diag(s_) @ Vt
    plt.imshow(A_, cmap='gray')
    plt.axis('off')
    plt.title("singular value: {}".format(s[i]))
    plt.show()

# 指定した特異値の割合までの行列を足して、元の画像を復元して可視化する
percent_sv = 0.999  # 99.9%までの特異値を使用
total_energy = np.sum(s ** 2)
energy = 0.0
num = 0

while energy / total_energy < percent_sv:
    energy += s[num] ** 2
    num += 1

s_ = np.zeros_like(s)
s_[:num] = s[:num]
A_ = U @ np.diag(s_) @ Vt
plt.imshow(A_, cmap='gray')
plt.axis('off')
plt.title("reconstructed image ({}% energy)".format(percent_sv * 100))
plt.show()

# 99.9%の画像を復元するために必要な特異値（画像）の個数
print("ここに99.9%の画像を復元するために必要な特異値（画像）の個数を出力")