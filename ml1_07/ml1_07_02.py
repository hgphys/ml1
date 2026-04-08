"""
データを読み込み主成分分析を行うプログラム
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# データの読み込み
df = pd.read_csv('正しくファイルパスを入力')

# 特徴量の抽出
X = df.values

# 主成分分析の実行
pca = PCA()
pca.fit(X)

# 主成分分析の実行
pca = PCA()
pca.fit(X)

# 第1主成分と第2主成分に射影
projected = pca.transform(X)
component1 = projected[:, 0]
component2 = projected[:, 1]

# 寄与率の取得
explained_variance_ratio = pca.explained_variance_ratio_
contribution1 = explained_variance_ratio[0]
contribution2 = 0 # ここに正しく入力

# 元の散布図の可視化
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2])
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_zlabel('Feature 3')
ax1.set_title('Original Data')

# 射影した散布図の可視化
ax2 = fig.add_subplot(122)
ax2.scatter(component1, component2)
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')
ax2.set_title('Projected Data')

# 寄与率の出力
print("第1主成分の寄与率: {:.2%}".format(contribution1))
print("第2主成分の寄与率: {:.2%}".format(contribution2))

# プロットの表示
plt.tight_layout()
plt.show()
