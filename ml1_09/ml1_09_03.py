"""
重回帰を行うクラスlinearreg.pyを用いて
読み込んだデータに対して重回帰モデルで学習するプログラム
"""

import linearreg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSVファイルの読み込み
df = pd.read_csv('data/training_data.csv')

# 特徴量行列の定義
X = np.array(df[['Poverty','Less than 9th grade']])

# 目的変数のデータ列の定義
y = np.array(df['Crime'])

# クラスを利用して重回帰
model = linearreg.LinearRegression()
model.fit(X, y)
print("w=", model.w_)

# 予測値と残差の追加
df['Crime_pred'] = model.predict(X)
df['Residual'] = df['Crime'] - df['Crime_pred']


# 回帰結果の可視化と評価
fig = plt.figure(figsize=(14,6))

# 回帰平面のプロット =========================== #
ax1 = fig.add_subplot(121, projection='3d')

# データ点のプロット
X1 = X.T[0]
X2 = X.T[1]
ax1.scatter(X1, X2, y, c='r', marker='o')

# 垂線のプロット
for i in range(len(df)):
    x_sample = np.array([[X1[i], X2[i]]])
    y_sample_pred = model.predict(x_sample)
    ax1.plot([X1[i], X1[i]], [X2[i], X2[i]], [y[i], 0], 'b--')

# 回帰平面のプロット
x1_grid, x2_grid = np.meshgrid(np.linspace(0, 20, 100), np.linspace(0, 20, 10))
X_grid = np.column_stack((x1_grid.flatten(), x2_grid.flatten()))
y_pred = model.predict(X_grid)
ax1.plot_surface(x1_grid, x2_grid, y_pred.reshape(x1_grid.shape), alpha=0.5)

# 軸ラベルの設定
ax1.set_xlabel('Poverty')
ax1.set_ylabel('Less than 9th grade')
ax1.set_zlabel('Crime')

# ========================================== #

# 残差プロット　  =========================== #
ax2 = fig.add_subplot(122)

ax2.scatter(df['Crime_pred'], df['Residual'], c = "r")
ax2.axhline(y=0, color='blue', linestyle='--')

# 軸ラベルの設定
ax2.set_xlabel('Prediction')
ax2.set_ylabel('Residual')

# ========================================== #
plt.show()

# 決定係数の計算
r2 = 1 - df['Residual'].var()/df['Crime'].var()
print("R2=",round(r2,3))