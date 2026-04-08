"""
Irisデータセットを読み込み、相関の強い特徴量を探索する
"""

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

# irisデータセットを読み込み
iris = load_iris()
X = iris.data
feature_names = iris.feature_names

# 相関行列を作成
corr_matrix = np.corrcoef(X.T)

# 相関行列ヒートマップで可視化
fig, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
            xticklabels=feature_names,
            yticklabels=feature_names, ax=ax)
plt.title("Correlation matrix of iris dataset")
plt.show()