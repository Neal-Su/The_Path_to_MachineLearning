# 特征预处理(StandardScaler标准化)

# 通过对原始数据进行变换把数据变换到均值为0,标准差为1范围内
# 对于归一化来说：如果出现异常点，影响了最大值和最小值，那么结果显然会发生改变
# 对于标准化来说：如果出现异常点，由于具有一定数据量，少量的异常点对于平均值的影响并不大，从而方差改变较小。

import pandas as pd
from sklearn.preprocessing import StandardScaler


def stand_demo():
    """
    标准化
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
    transfer = StandardScaler()

    # 3.调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    return None


if __name__ == '__main__':
    # 代码8：标准化
    stand_demo()
