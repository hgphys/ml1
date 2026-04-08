"""
四則演算とif文、配列の復習
以下のコードは未完成です。
"""

#児童・生徒のデータ（身長[cm], 体重[kg]）
children = [
    {'name': 'A', 'height': 130, 'weight': 38},
    {'name': 'B', 'height': 140, 'weight': 42},
    {'name': 'C', 'height': 145, 'weight': 30},
    {'name': 'D', 'height': 150, 'weight': 35},
    {'name': 'E', 'height': 160, 'weight': 48}
]

#標準に分類されたローレル指数のリスト
rohrer_index_standard = []

for child in children:
    name = child['name']
    height = child['height']
    weight = child['weight']
    rohrer =  #ローレル指数の定義を記載しましょう
    print(name, height, weight, rohrer)
    """
    ここに rohrer が標準である場合、
    rohrer_index_standard にその値を追加するプログラムを記載しましょう
    """


#リストの長さを出力
print("標準に分類された人数：", len(rohrer_index_standard))

#標準に分類された児童・生徒のローレル指数の最大値の計算
max_rohrer_index_standard = max(rohrer_index_standard)

#結果の出力（小数点第2位を四捨五入）
print("標準に分類されたローレル指数の最大値：", round(max_rohrer_index_standard,1))