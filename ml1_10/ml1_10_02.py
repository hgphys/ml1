"""
多項式回帰を行うクラスpolyregを用いて
読み込んだデータの多項式回帰を行うプログラム
"""

import polyreg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSVファイルの読み込み
df = pd.read_csv('data/data01.csv')

# 訓練データ 
x_training = np.array(df['x'])
y_training = np.array(df['y'])


# クラスを利用して多項式回帰
list_L2w = []
list_d = list(range(1, 13))
for d in list_d:
    modeli = polyreg.PolynomialRegression(d)
    modeli.fit(x_training, y_training)
    L2wi = np.linalg.norm(modeli.w_, ord=2) # パラメータのL2ノルムの計算
    list_L2w.append(L2wi)

# パラメータのL2ノルムの次元数の関係プロット
fig = plt.figure(figsize=(8,5))
plt.plot(list_d, list_L2w, 'r-o')
plt.xlabel('Degree $d$', fontsize = 15)
plt.ylabel('L2 norm $\|w\|_2$', fontsize = 15)
plt.show()

# d=8のL2ノルムの値の出力
print("ここにd=8のL2ノルムの値を出力する")