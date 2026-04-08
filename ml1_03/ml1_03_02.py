"""
Pandasと関数の利用
BMIとBMIに基づく肥満度の列を関数を用いて追加する
"""

import pandas as pd
from mod03 import calculate_bmi, judge_obesity_bmi

# サンプルデータの読み込み
df = pd.read_csv("data.csv")

# BMIと肥満度の列を追加
df['BMI'] = df.apply(lambda row: calculate_bmi(row['身長[cm]'], row['体重[kg]']), axis=1)
df['肥満度（BMI）'] = df['BMI'].apply(lambda x: judge_obesity_bmi(x))

# 肥満度（BMI）の分布
print("ここに頻度を出力させる")

# CSVファイルに出力
df.to_csv("data02.csv", index=False)