"""
正則化項を含む多項式回帰を行うクラスpolyridgeを用いて
読み込んだデータの多項式回帰を行うプログラム
"""

import polyridge
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSVファイルの読み込み
df = pd.read_csv('data/data03.csv')

# 訓練データ 偶数行
x_training = np.array(df['x'][::2]) 
y_training = np.array(df['y'][::2]) 

# テストデータ　　奇数行
x_test = np.array(df['x'][1::2]) 
y_test = np.array(df['y'][1::2])


# クラスを利用して多項式回帰 d=9 alpha
alpha = 0
model9 = polyridge.PolyRidgeRegression(9, alpha)
model9.fit(x_training, y_training)

# データと回帰曲線のプロット
fig = plt.figure(figsize=(8,5))
plt.scatter(x_training, y_training, c='r', marker='o', label = "Training Data")
plt.scatter(x_test, y_test, c='g', marker='o', label = "Test Data")
plt.ylim([y_training.min()-1, y_test.max()+1])

xx = np.linspace(x_training.min(), x_test.max(), 300)
y9 = np.array([model9.predict(u) for u in xx])

plt.plot(xx, y9, color = 'b', label=r'$d=9, \alpha =$' + str(alpha))

plt.xlabel('x', fontsize = 15)
plt.ylabel('y', fontsize = 15)
plt.legend(fontsize=15)
plt.show()

# テストデータに対する残差の計算
y9_pred = model9.predict(x_test)
residual9 = y_test - y9_pred
print("テストデータに対する残差の二乗平均", round(np.mean(residual9**2),2))