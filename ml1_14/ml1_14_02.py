"""
２次元入力3クラス分類モデルの実装
logisticreg3.pyの利用
"""

import numpy as np
import pandas as pd
from logisticreg3 import LogisticRegression
from sklearn.metrics import confusion_matrix

# CSVファイル読み込み
df = pd.read_csv("data/data.csv")

# 特徴量とラベルに分割
X = np.array(df[["Feature 1", "Feature 2"]])
y = df["Label"].values

# 偶数行を訓練データと奇数行をテストデータに分割
X_train, X_test = X[::2], X[1::2]
y_train, y_test = y[::2], y[1::2]

# 訓練データの正解ラベルをone-hotエンコーディング
y_train_encoded = pd.get_dummies(y_train)
T_train = np.array(y_train_encoded)

# モデルの学習
model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
weights, cost = model.fit(X_train, T_train)


# テストデータの予測
y_pred = model.predict(X_test)

# ラベルの変換
labels = np.unique(y)
print(labels)
label_dict = {label: i for i, label in enumerate(labels)}
print(label_dict)
y_test_num = np.array([label_dict[label] for label in y_test])

# 混同行列の計算
confusion = confusion_matrix(y_test_num, y_pred)

# 混同行列の表示
print("Confusion Matrix:")
print(confusion)

# 正解率の計算
accuracy = 0 # ここに正解率を正しく定義する
print(f"正解率: {accuracy:.2f}%")