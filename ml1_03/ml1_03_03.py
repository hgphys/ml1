"""
Pandasと関数、の利用
BMIとBMIに基づく肥満度の列を関数を用いて追加する
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from mod03 import judge_obesity_bodyfat

# サンプルデータの読み込み
df = pd.read_csv("data02.csv")

# 体脂肪率に基づく肥満度の列を追加
df['肥満度（体脂肪率）'] = df['体脂肪率[%]'].apply(lambda x: judge_obesity_bodyfat(x))

# CSVファイルに出力
df.to_csv("data03.csv", index=False)

# CSVファイルを読み込み
df_read = pd.read_csv('data03.csv')

# 正解列と予測列を取得
y_true = df_read['肥満度（体脂肪率）']
y_pred = df_read['肥満度（BMI）']

# 正解率を計算
accuracy = accuracy_score(y_true, y_pred)

# 結果を出力
print("正解率: {:.2f}".format(accuracy))