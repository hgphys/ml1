"""
１次元入力２クラス分類に関する
サンプルデータを読み込み条件付き確率を確認するプログラム
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データの読み込み
df = pd.read_csv('data/data.csv')
X = df['X'].values
y = df['y'].values

# データ点の色
X_col=['cornflowerblue','gray']

# データの可視化
plt.figure(figsize=(5,5))
plt.scatter(X[y == 0], y[y == 0], color=X_col[0], label='y = 0')
plt.scatter(X[y == 1], y[y == 1], color=X_col[1], label='y = 1')

# y=1の最小値となるXの値を取得
X_min_y1 = X[y == 1].min()
# y=0が最大となるXの値を取得
X_max_y0 = X[y == 0].max()

# 縦線を追加
plt.axvline(X_min_y1, color=X_col[1], linestyle='--', label='X='+str(round(X_min_y1,1)))
plt.axvline(X_max_y0, color=X_col[0], linestyle='--', label='X=' +str(round(X_max_y0,1)))

plt.xlabel('X',fontsize=15)
plt.ylabel('y',fontsize=15)
plt.legend(loc='lower right',fontsize=15)
plt.grid(True) 
plt.show()

# y=1とy=0のデータ点の個数を取得
count_between = ((X >= X_min_y1) & (X <= X_max_y0)).sum()
count_y1_between = ((X >= X_min_y1) & (X <= X_max_y0) & (y == 1)).sum()

# 結果の出力
print("y=1のうち最小のX:", round(X_min_y1,1))
print("y=0のうち最大のX:", round(X_max_y0,1))
print(str(round(X_min_y1,1))+'≦X≦' +str(round(X_max_y0,1)) + "の間のデータの個数:", count_between)
print("このうちy=1であるデータの個数:", count_y1_between)

# P(y=1|0.9 <= <= 1.2) の場合の条件付き確率の計算
print(round(count_y1_between/count_between,2))