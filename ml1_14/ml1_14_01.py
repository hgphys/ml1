"""
２次元入力3クラス分類モデルの実装
logisticreg3.pyの利用
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from logisticreg3 import LogisticRegression

# CSVファイル読み込み
df = pd.read_csv("data/data.csv")

# 特徴量とラベルに分割
X = df[["Feature 1", "Feature 2"]]
y = df["Label"]

# 正解ラベルをone-hotエンコーディング
y_encoded = pd.get_dummies(y)
T = np.array(y_encoded)

# モデルの学習
model = LogisticRegression(learning_rate=0.001, num_iterations=1000)
weights, cost = model.fit(X, T)

# 重み行列と平均交差エントロピー誤差の出力
print("Weights:\n", weights)
print("Average Cross-Entropy Loss:", cost)

# サンプルデータの散布図と決定境界の可視化
h = 0.02  # メッシュのステップサイズ
x_min, x_max = X["Feature 1"].min() - 1, X["Feature 1"].max() + 1
y_min, y_max = X["Feature 2"].min() - 1, X["Feature 2"].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(6, 6))
plt.contour(xx, yy, Z, levels=np.arange(np.unique(y).shape[0] + 1) - 0.5, colors='grey')
color_dict = dict({"A":'red',
                   "B":'green',
                   "C":'blue'})
sns.scatterplot(x="Feature 1", y="Feature 2", hue="Label", data=df, palette=color_dict)
plt.xlabel('Feature 1', fontsize = 15)
plt.ylabel('Feature 2', fontsize = 15)
plt.show()


# データの3次元プロット
fig = plt.figure(figsize=(12, 4))
labels = np.unique(y)
colors = ['r', 'g', 'b']

# ソフトマックス関数の出力を計算
Xtilde = np.c_[np.ones((len(df), 1)), X]  # バイアス項を追加
z = np.dot(Xtilde, weights)
probabilities = model.softmax(z)

for i, label in enumerate(labels):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    
    indices_ = np.where(y == label)
    for j, label_ in enumerate(labels):
        indices = df["Label"] == label_
        ax.scatter(X[indices]['Feature 1'], X[indices]['Feature 2'], probabilities[indices_, j], c=colors[j])
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'Class {label}')

plt.tight_layout()
plt.show()