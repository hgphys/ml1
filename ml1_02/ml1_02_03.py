"""
Numpy, Pandas の利用準備、配列の復習
"""

import numpy as np
import pandas as pd

# データの生成
height = np.random.normal(loc=170, scale=5, size=100)  # 平均170cm、標準偏差5cmの正規分布からランダムサンプリング
weight = np.random.normal(loc=60, scale=10, size=100)  # 平均60kg、標準偏差10kgの正規分布からランダムサンプリング
body_fat = np.random.normal(loc=15, scale=5, size=100)  # 平均15%、標準偏差5%の正規分布からランダムサンプリング

# データの結合
df = pd.DataFrame({"身長[cm]": height, "体重[kg]": weight, "体脂肪率[%]": body_fat})

# CSVファイルに出力
df.to_csv("sample_data.csv", index=False)

# データの確認
df_read = pd.read_csv("sample_data.csv")
print(df_read.head())