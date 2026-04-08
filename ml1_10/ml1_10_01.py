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

# 訓練データ 偶数行
x_training = np.array(df['x']) # ここを正しく記載
y_training = np.array(df['y']) # ここを正しく記載

# テストデータ　　奇数行
x_test = np.array(df['x']) # ここを正しく記載
y_test = np.array(df['y']) # ここを正しく記載

# クラスを利用して多項式回帰 d=1 (単回帰モデル)
model1 = polyreg.PolynomialRegression(1)
model1.fit(x_training, y_training)

# クラスを利用して多項式回帰 d=9
model9 = polyreg.PolynomialRegression(9)
model9.fit(x_training, y_training)
print(model9.w_)

# データと回帰曲線のプロット
fig = plt.figure(figsize=(8,5))
plt.scatter(x_training, y_training, c='r', marker='o', label = "Training Data")
plt.scatter(x_test, y_test, c='g', marker='o', label = "Test Data")
plt.ylim([y_training.min()-1, y_test.max()+1])

xx = np.linspace(x_training.min(), x_test.max(), 300)
y1 = np.array([model1.predict(u) for u in xx])
y9 = np.array([model9.predict(u) for u in xx])

plt.plot(xx, y1, color = 'r', linestyle = 'dashed', label='1st degree')
plt.plot(xx, y9, color = 'b', label='9th degree')

plt.xlabel('x', fontsize = 15)
plt.ylabel('y', fontsize = 15)
plt.legend(fontsize=15)
plt.show()

# テストデータに対する残差の計算
y1_pred = model1.predict(x_test)
residual1 = y_test - y1_pred
print("テストデータに対する残差の二乗平均(d=1)", round(np.mean(residual1**2),2))
y9_pred = model9.predict(x_test)
residual9 = y_test - y9_pred
print("テストデータに対する残差の二乗平均(d=9)", round(np.mean(residual9**2),2))