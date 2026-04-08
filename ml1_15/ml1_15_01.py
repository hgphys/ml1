"""
リッジ回帰のクラスridge.pyを用いて
読み込んだデータに対してリッジ回帰モデルで学習するプログラム
"""

import ridge
import pandas as pd
import numpy as np

# CSVファイルの読み込み
df = 

# 特徴量行列と目的変数の定義
X = 
y = 

# データを偶数行の訓練データと奇数行のテストデータに分割
X_train, y_train = 
X_test, y_test =

# クラスを利用してリッジ回帰
alpha = 1.0
model = 

print("alpa=", alpha)
print("w=", model.w_)

# テストデータで予測
y_pred_train = 
y_pred_test = 

# 平均二乗誤差と決定係数を評価
mse_train = np.mean((y_train - y_pred_train) ** 2)
r2_train = 1 - np.mean((y_train - y_pred_train) ** 2) / np.var(y_train)

mse_test = np.mean((y_test - y_pred_test) ** 2)
r2_test = 1 - np.mean((y_test - y_pred_test) ** 2) / np.var(y_test)

print("訓練データの平均二乗誤差 (MSE): {:.2f}".format(mse_train))
print("訓練データの決定係数 (R^2): {:.3f}".format(r2_train))

print("テストデータの平均二乗誤差 (MSE): {:.2f}".format(mse_test))
print("テストデータの決定係数 (R^2): {:.3f}".format(r2_test))