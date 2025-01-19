# 特征降维(主成分分析PCA降维)

# 主成分分析
# 定义：高维数据转化为低维数据的过程，在此过程中可能会舍弃原有数据、创造新的变量
# 作用：是数据维数压缩，尽可能降低原数据的维数（复杂度），损失少量信息。
# 应用：回归分析或者聚类分析当中

from sklearn.decomposition import PCA


def pca_demo():
    """
    PCA降维
    :return:
    """
    data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]
    # 1.实例一个转换器类
    # sklearn.decomposition.PCA(n_components=None)
    # 将数据分解为较低维数空间
    # n_components:
    # 小数：表示保留百分之多少的信息
    # 整数：减少到多少个特征
    transfer = PCA(n_components=3)

    # 2.调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)

    return None


if __name__ == '__main__':
    # 代码11：PCA降维
    pca_demo()
