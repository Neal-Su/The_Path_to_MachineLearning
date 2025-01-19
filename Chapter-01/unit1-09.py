# 特征降维(特征选择-过滤式-方差选择法)

# Filter(过滤式)：主要探究特征本身特点、特征与特征和目标值之间关联
# 方差选择法：低方差特征过滤
# 相关系数法

# 方差选择法：低方差特征过滤
# 删除低方差的一些特征，前面讲过方差的意义。再结合方差的大小来考虑这个方式的角度。
# 特征方差小：某个特征大多样本的值比较相近
# 特征方差大：某个特征很多样本的值都有差别

import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def variance_demo():
    """
    过滤低方差特征
    :return:
    """
    # 1.获取数据
    # data_path设置路径
    data_path = r"E:\B python wenjian\factor_returns.csv"
    # 使用pandas库读取txt文件
    data = pd.read_csv(data_path)
    data = data.iloc[:, 1:-2]  # pandas库里的一个功能，取第2列至倒数第3列保留
    print("删除前形状：\n", data.shape)

    # 2.实例一个转换器类
    # threshold=5，代表的是删除掉方差小于5的特征
    transfer = VarianceThreshold(threshold=5)

    # 3.调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    print("删除后形状：\n", data_new.shape)

    return None


if __name__ == '__main__':
    # 代码9：过滤低方差特征
    variance_demo()