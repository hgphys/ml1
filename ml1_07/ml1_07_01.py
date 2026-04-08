"""
CSVファイルを読み込み統計基本量を計算するプログラム
"""

import pandas as pd
import numpy as np

# 金沢のCSVファイルから日別平均気温を読み込む
kanazawa_data = pd.read_csv('正しくファイルパスを入力')
kanazawa_temps = kanazawa_data['平均気温(℃)'].values

# 東京のCSVファイルから日別平均気温を読み込む
tokyo_data = pd.read_csv('正しくファイルパスを入力')
tokyo_temps = tokyo_data['平均気温(℃)'].values

# 平均と標準偏差を計算
tokyo_mean = np.mean(tokyo_temps)
tokyo_std = np.std(tokyo_temps)
kanazawa_mean = 0 # ここに正しく入力
kanazawa_std = 0 # ここに正しく入力

# 相関係数を計算
correlation = np.corrcoef(kanazawa_temps, tokyo_temps)[0, 1]

# 結果を出力
print("金沢の平均気温: {:.2f}".format(kanazawa_mean))
print("金沢の標準偏差: {:.2f}".format(kanazawa_std))
print("金沢と東京の気温の相関係数: {:.2f}".format(correlation))
