"""
CSVファイルからデータを読み込み散布図として可視化
このデータを自分自身で平均二乗誤差を指針に直線でフィットする
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mod04 import function

# CSVファイルからデータを読み込む
data = pd.read_csv('data03.csv')

# データをxとyに分割
x = data['x']
y = data['y']

# 散布図をプロット
plt.scatter(x, y, c='b', alpha=0.5, label='data03')

# 予測直線のパラメータを指定（ただし、a,bは整数）
a = 1
b = 1

# 平均二乗誤差（MSE）の計算
mse = np.mean((function(x, a, b) - y)**2)

# 予測直線をプロット
plt.plot(x, function(x, a, b), 'r', label='prediction: a={0}, b={1}'.format(a,b))

# グラフのタイトルや軸ラベルを設定
plt.title('MSE = {:.2f}'.format(mse))
plt.xlabel('x')
plt.ylabel('y')

# 凡例を追加
plt.legend()

# グラフを表示
plt.show()