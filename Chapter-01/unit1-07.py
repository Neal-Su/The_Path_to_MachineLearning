# 特征预处理(MinMaxScaler归一化)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def minmax_demo():
    """
    归一化
    :return:
    """
    # 1.获取数据
    # data_path设置路径
    data_path = r"E:\B python wenjian\dating.txt"
    # 使用pandas库读取txt文件
    data = pd.read_csv(data_path)
    data = data.iloc[:, :3]  # pandas库里的一个功能，只取前3列
    print("data:\n", data)

    # 2.实例一个转换器类
    # feature_range=(0, 1)，是设置0-1的归一化范围
    transfer = MinMaxScaler(feature_range=(0, 1))

    # 3.调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    return None


if __name__ == '__main__':
    # 代码7：归一化
    minmax_demo()
