"""
以下の関数には誤りが含まれます
"""

def function(x, a = 2, b = 1):
    """
    １次関数
    :param x: 入力
    :param a: 傾き
    :param b: 切片
    :return: ax + b
    """
    return x


def calculate_bmi(height: float, weight: float) -> float:
    """
    BMIを計算する関数
    :param height: 身長(cm)
    :param weight: 体重(kg)
    :return: BMI
    """
    bmi = weight / ((height/100) ** 2)
    return round(bmi, 2)


def judge_obesity_bmi(bmi: float) -> str:
    """
    BMIから肥満度を判定する関数
    :param bmi: BMI
    :return: 肥満度の判定結果
    """
    if bmi < 18.5:
        return "低体重（やせ）"
    elif bmi < 25:
        return "普通体重"
    else:
        return "肥満"


def judge_obesity_bodyfat(bodyfat: float) -> str:
    """
    体脂肪率から肥満度を判定する関数
    :param bmi: 体脂肪率(%)
    :return: 肥満度の判定結果
    """
    if bodyfat < 15:
        return "低体重（やせ）"
    elif bodyfat < 20:
        return "普通体重"
    else:
        return "肥満"