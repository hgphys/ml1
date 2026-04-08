"""
暴力事件件数の分析用データの前処理を行うプログラム
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 州別の10万人あたり暴力事件件数データのエクセルファイルの読み込みと加工
df_crime = pd.read_excel('data/statistic_id200445_reported-violent-crime-rate-in-the-us-2020-by-state.xlsx',
                         sheet_name='Data', skiprows = 5, usecols = [1,2], names = ['State', 'Crime'], header = None)

# 州別の貧困率データのCSVファイルの読み込みと加工
df_poverty = pd.read_csv('ダウンロードしたファイルのパスを入力')

list_col_proverty = list(df_poverty.columns)[5:][::6] # 必要な列名のリスト作成
new_col_names = [list_col_proverty[i].split('!!')[0] for i in range(len(list_col_proverty))] # 州名リストの作成
col_names_dict = dict(zip(list_col_proverty, new_col_names)) # 列名変更のための辞書の作成
df_poverty = df_poverty[list_col_proverty][:1].rename(columns=col_names_dict) # 州名として列名変更
df_poverty = df_poverty.T.rename(columns = {0:'Poverty'}).reset_index() # 行列を説明変数の名前を再定義
df_poverty = df_poverty.rename(columns = {'index':'State'})


# 州別の教育水準データのCSVファイルの読み込みと加工
df_education = pd.read_csv('ダウンロードしたファイルのパスを入力')

list_col_education = list(df_education.columns)[3:][::12] # 必要な列名のリスト作成
new_col_names = [list_col_education[i].split('!!')[0] for i in range(len(list_col_education))] # 州名リストの作成
col_names_dict = dict(zip(list_col_education, new_col_names)) # 列名変更のための辞書の作成
df_education = df_education[list_col_education].rename(columns=col_names_dict) # 州名として列名変更
df_education = df_education[7:8] # 25歳以上の中卒割合の行を抽出

df_education = df_education.T.rename(columns = {7:'Less than 9th grade'}).reset_index() # 行列を説明変数の名前を再定義
df_education = df_education.rename(columns = {'index':'State'})


# ３つのデータを州名で結合してデータタイプの修正
df = pd.merge(df_crime, df_poverty, on ='State')
df = pd.merge(df, df_education, on ='State')
df['Poverty'] = df['Poverty'].str.replace('%', '').astype(float)
df['Less than 9th grade'] = df['Less than 9th grade'].str.replace('%', '').astype(float)

# データセットの保存
df.to_csv('data/training_data.csv', index=False)

# 保存データの確認
df = pd.read_csv('data/training_data.csv')
print(df.head())


# データセットの可視化
fig = plt.figure(figsize = (7,7))
ax = fig.add_subplot(111, projection='3d')

X1 = df['Poverty']
X2 = df['Less than 9th grade']
y = df['Crime']

# データ点のプロット
ax.scatter(X1, X2, y, c='r', marker='o')

# 垂線のプロット
for i in range(len(df)):
    ax.plot([X1[i], X1[i]], [X2[i], X2[i]], [y[i], 0], 'b--')

# 軸ラベルの設定
ax.set_xlabel('Poverty')
ax.set_ylabel('Less than 9th grade')
ax.set_zlabel('Crime')

plt.show()

# 相関行列の表示
print(df[['Crime','Poverty','Less than 9th grade']].corr())