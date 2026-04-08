"""
２次元入力２クラス分類モデルの実装
logisticreg.pyの利用
"""

import numpy as np
import pandas as pd
import logisticreg
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# データの読み込み
df = pd.read_csv('data/data.csv')
X = np.array(df[['X1', 'X2']])
y = df['y'].values

# 偶数行を訓練データと奇数行をテストデータに分割
X_train, X_test = X[::2], X[1::2]
y_train, y_test = y[::2], y[1::2]

# ロジスティック回帰の学習
model = logisticreg.LogisticRegression(tol=0.01)
model.fit(X_train, y_train)

# テストデータに対する予測
y_pred = model.predict(X_test)

# 混同行列の計算
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion)

# 正解率の計算
accuracy = (confusion[0,0] + confusion[1,1])/ confusion.sum() #正解率の定義
print("Accuracy:", accuracy)

# 適合率の計算
precision = confusion[0,0] / (confusion[0,0]+confusion[0,1])  #適合率の定義
print("Precision:", precision)

# 再現率の計算
recall = confusion[0,0] / (confusion[0,0]+confusion[1,0])  #再現率の定義
print("Recall:", recall)