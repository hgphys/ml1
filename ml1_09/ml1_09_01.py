"""
重回帰を行うクラスlinearreg.pyを用いて
読み込んだデータの２次元回帰を行うプログラム
"""

import linearreg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSVファイルの読み込み
df = pd.read_csv('data/data01.csv')

# 特徴量行列の定義
X = np.array(df[['x1','x2']])

# 目的変数のデータ列の定義
y = np.array(df['y'])

# クラスを利用して重回帰
model = linearreg.LinearRegression()
model.fit(X, y)
print("w=", model.w_)


# 回帰平面のプロット
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

# データ点のプロット
X1 = X.T[0]
X2 = X.T[1]
ax.scatter(X1, X2, y, c='r', marker='o')

# 垂線のプロット
for i in range(len(df)):
    x_sample = np.array([[X1[i], X2[i]]])
    y_sample_pred = model.predict(x_sample)
    ax.plot([X1[i], X1[i]], [X2[i], X2[i]], [y[i], 0], 'b--')

# 回帰平面のプロット
x1_grid, x2_grid = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 10, 10))
X_grid = np.column_stack((x1_grid.flatten(), x2_grid.flatten()))
y_pred = model.predict(X_grid)
ax.plot_surface(x1_grid, x2_grid, y_pred.reshape(x1_grid.shape), alpha=0.5)

# 軸ラベルの設定
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')

plt.show()