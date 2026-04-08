"""
CSVファイルを読み込み、データを分割して保存、
統計基本量を計算するプログラム
"""

import numpy as np
import pandas as pd

# CSVファイルの読み込みと分割
df = pd.read_csv('data/data.csv')

# 訓練データとテストデータに分割、保存
training_data = df # dfの最初の70個 修正が必要
test_data = df # dfの70番目以降　修正が必要

training_data.to_csv('data/training_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)

# 分散と共分散の計算
variance = training_data['x'].var()
covariance = training_data.cov()

print("Var[x]:", round(variance,2))
print("Cov[x,y]:", round(covariance.at['x','y'],2))
