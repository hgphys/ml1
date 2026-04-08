"""
単純パーセプトロンの入力と出力
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データの読み込み
df = pd.read_csv('data/data.csv')
X = df['X'].values
y = df['y'].values

# 重みの定義
input_weights = np.array([-5, 5])

# 平均交差エントロピー誤差の定義
def cross_entropy_loss(X, y, weight):
    ones = np.ones(X.shape[0])
    Xtild = np.column_stack((ones, X))  # バイアス項を追加
    yhat = 1 / (1 + np.exp(-np.dot(Xtild, weight)))  # シグモイド関数
    loss = -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))  # 平均交差エントロピー誤差

    return loss

# 与えた重みのときの平均交差エントロピー誤差の出力
print("w0=",input_weights[0])
print("w1=",input_weights[1])
print("E(w0,w1)=", round(cross_entropy_loss(X, y, input_weights ),2))

# 交差エントロピー誤差の等高線プロット
w0_vals = np.linspace(-15, 0, 100)
w1_vals = np.linspace(0, 15, 100)
w0_mesh, w1_mesh = np.meshgrid(w0_vals, w1_vals)
loss_vals = np.zeros((100, 100))

for i in range(100):
    for j in range(100):
        weight = np.array([w0_mesh[i, j], w1_mesh[i, j]])
        loss = cross_entropy_loss(X, y, weight)
        loss_vals[i, j] = loss


plt.contour(w0_mesh, w1_mesh, loss_vals, levels=50, cmap='viridis')
plt.colorbar()
plt.scatter(input_weights[0],input_weights[1], label = "input_weight")
plt.xlabel('w0',fontsize=15)
plt.ylabel('w1',fontsize=15)
plt.legend(fontsize=15)
plt.show()