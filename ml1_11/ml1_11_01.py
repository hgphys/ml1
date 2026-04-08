"""
Scikitlearnを用いてポアソン回帰を行うプログラム
"""

import pandas as pd
from sklearn.linear_model import PoissonRegressor
import numpy as np

# データの読み込み
data = pd.read_csv('data/data01.csv') 

# 説明変数と目的変数の指定
X = data[['math', 'path_ac', 'path_vo']]
y = data['awards']

# ポアソン回帰モデルの構築と学習
model = PoissonRegressor()
model.fit(X, y)

# 重みの出力
weights = model.coef_
intercept = model.intercept_

print("Weights:", weights)
print("Intercept:", intercept)

print(np.exp(weights[0])) # w1
print(np.exp(weights[1])) # w2