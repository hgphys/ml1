"""
ロジスティック回帰のクラスlogisticreg.pyを用いて
読み込んだデータに対してロジスティック回帰モデルで学習するプログラム
"""

import numpy as np
import pandas as pd
from logisticreg import LogisticRegression
from sklearn.metrics import confusion_matrix

# CSVファイル読み込み
df = 

# 特徴量行列と目的変数の定義
X = 
y = 

# ラベルを数値に変換（各品種を0, 1, 2に対応する数値に変換する）
species_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
y_encoded = 

# 偶数行を訓練データと奇数行をテストデータに分割
X_train, y_train = 
X_test, y_test = 

# 訓練データの正解ラベルをone-hotエンコーディング
y_train_encoded = 
T_train = 

# モデルの学習
model = LogisticRegression(learning_rate=1.0, num_iterations=10000)
weights, cost = model.fit(X_train, T_train)

# テストデータの予測
y_pred = model.predict(X_test)

# 混同行列の計算
confusion = confusion_matrix(y_test, y_pred)

# 混同行列の表示
print("Confusion Matrix:")
print(confusion)

# 正解率の計算
accuracy = np.trace(confusion) /  np.sum(confusion)  * 100 # ここに正解率を定義する
print(f"正解率: {accuracy:.1f}%")

# 適合率の計算
precision = np.diag(confusion) / np.sum(confusion, axis=0) * 100 # ここに正解率を定義する
print("適合率:")
print(precision)