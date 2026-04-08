"""
訓練データを読み込み勾配降下法によって
単回帰のパラメータ推定を行うプログラム
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 訓練データの読み込み
training_data = pd.read_csv('data/training_data.csv')

# ハイパーパラメータの定義
max_epochs = 10000 #最大繰り返し回数
alpha = 0.01 #学習率
eps = 1e-4 #収束条件

# モデルを行列表現するための定義
Xtil = np.vstack([np.ones_like(training_data['x']), training_data['x']]).T
y = np.array(training_data['y'])

# パラメータの初期化
w = np.zeros(Xtil.shape[1])

# 勾配降下法の定義
for t in range(max_epochs):
    y_hat = Xtil @ w 
    grad = 0 # ここを正しく入力する
    if np.sum(np.abs(grad)) < eps:
        break
    w -= alpha * grad

# 結果の出力
print("w0 = ", round(w[0],2))
print("w1 = ", round(w[1],2))

# パラメータの保存（CSV形式）
model_params = pd.DataFrame({'w0': [w[0]], 'w1': [w[1]]})
model_params.to_csv('model_params.csv', index=False)

# 回帰直線付きの散布図の可視化
plt.figure(figsize=(6, 6))
plt.scatter(training_data['x'], training_data['y'], label='Training Data')
plt.plot(training_data['x'], w[0] + w[1] * training_data['x'], color='red',
         label=f'Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression')
plt.grid(True)
plt.show()
