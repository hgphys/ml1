"""
２次元入力２クラス分類モデルの実装
logisticreg.pyの利用
"""

import numpy as np
import pandas as pd
import logisticreg
import matplotlib.pyplot as plt

# データの読み込み
df = pd.read_csv('data/data.csv')
X = np.array(df[['X1', 'X2']])
X1 = df['X1'].values
X2 = df['X2'].values
y = df['y'].values

# ロジスティック回帰の学習
model = logisticreg.LogisticRegression(tol=0.01)
model.fit(X, y)

# 重みの値を出力
w_opt = model.w_
print("重み:", w_opt)

# 分類結果の出力
yhat = model.predict(X)
print("ここにClass 1 に分類されたデータの個数を出力する")

# 予測結果の計算
X_ = np.linspace(np.min(X), np.max(X), 100)
Xtild_ = np.hstack((np.ones((len(X_), 2)), X_.reshape(-1, 1))) 
y_pred_value = logisticreg.sigmoid(np.dot(Xtild_, w_opt))

# メッシュグリッドの作成
x1_min, x1_max = X1.min() - 1, X1.max() + 1
x2_min, x2_max = X2.min() - 1, X2.max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                       np.arange(x2_min, x2_max, 0.1))
grid = np.c_[xx1.ravel(), xx2.ravel()]
grid = np.hstack((np.ones((grid.shape[0], 1)), grid))

# 予測結果の計算
y_pred_v = logisticreg.sigmoid(np.dot(grid, w_opt))
y_pred_value = y_pred_v.reshape(xx1.shape)

# 回帰曲面とデータ点の3次元プロット
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1[y == 1], X2[y == 1], y[y == 1], c='red', label='Class 1') # クラス1のデータ点のプロット
ax.scatter(X1[y == 0], X2[y == 0], y[y == 0], c='blue', label='Class 0') # クラス0のデータ点のプロット
ax.plot_surface(xx1, xx2, y_pred_value, alpha=0.5, cmap='jet') # 回帰曲面のプロット
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
ax.legend()
ax.grid(True)
plt.show()


# 2次元データの散布図と決定境界の可視化
fig = plt.figure(figsize=(6,6))
plt.scatter(X1[y == 0], X2[y == 0], color='blue', label='Class 0')
plt.scatter(X1[y == 1], X2[y == 1], color='red', label='Class 1')
cont = plt.contour(xx1, xx2, y_pred_value, levels=[0.2, 0.5, 0.8], colors=['blue','black','red'],
            linestyles=['dashed', 'solid', 'dashed'])
cont.clabel(fmt='%1.1f', fontsize=12)
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.show()