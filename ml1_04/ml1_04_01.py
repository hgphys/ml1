"""
Irisデータセットを読み込み、種類別で色分けした散布図として可視化する
"""

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# irisデータセットを読み込み特徴量の相関関係を散布図で可視化
sns.pairplot(sns.load_dataset('iris'), hue='species')
