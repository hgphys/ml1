"""
２層パーセプトロンの入力と出力
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# シグモイド関数
def sigmoid(x):
    return   # ここにシグモイド関数を定義する

# ロジスティック回帰の平均交差エントロピー誤差
def logistic_loss(w, X, y):
    Xtild = np.hstack((np.ones((len(X), 1)), X.reshape(-1, 1))) 
    yhat = sigmoid(np.dot(Xtild, w))
    loss =  -np.mean( y * np.log(yhat) + (1 - y) * np.log(1 - yhat))  
    return loss

# ロジスティック回帰の学習
def gradient_descent(X, y, weight, alpha, num_iterations, eps):
    Xtild = np.hstack((np.ones((len(X), 1)), X.reshape(-1, 1))) 
    E_history = []
    for _ in range(num_iterations):
        yhat = sigmoid(Xtild.dot(weight))
        grad =  # ここに平均交差エントロピーの勾配を定義する
        weight -= (alpha) * grad
        E_history.append(logistic_loss(weight, X, y))
        if (grad**2).sum() < eps**2:
                break
    return weight, E_history

# データの読み込み
df = pd.read_csv('data/data.csv')
X = df['X'].values
y = df['y'].values

# パラメータの初期化
weight = np.array([0.,0.])

# ハイパーパラメータ
alpha = 0.1
num_iterations = 10000
eps = 1e-6

# ロジスティック回帰の学習
w_opt, E_history = gradient_descent(X, y, weight, alpha, num_iterations, eps)

# 重みの値を出力
print("重み:", w_opt)


# 予測結果の計算
X_ = np.linspace(np.min(X), np.max(X), 100)
Xtild_ = np.hstack((np.ones((len(X_), 1)), X_.reshape(-1, 1))) 
y_pred = sigmoid(np.dot(Xtild_, w_opt))

# データと回帰曲線の可視化

X_col=['cornflowerblue','gray'] # データ点の色
plt.scatter(X[y == 0], y[y == 0], color=X_col[0], label='y = 0')
plt.scatter(X[y == 1], y[y == 1], color=X_col[1], label='y = 1')
plt.plot(X_, y_pred, color='red', label='Logistic Regression')
plt.xlabel('X',fontsize=15)
plt.ylabel('y',fontsize=15)
plt.legend(loc='lower right',fontsize=15)
plt.grid(True)
plt.show()
