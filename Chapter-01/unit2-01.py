# KNN算法,以对iris分类预测为例

# K-近邻算法(KNN算法)
# 定义：
# 如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，
# 则该样本也属于这个类别。

# k值过小容易受到异常值影响。当样本不均衡的时候，k值过大可能会产生显著错误
# 如何确定距离：计算距离：欧式距离/曼哈顿距离(绝对值距离)/明可夫斯基距离

# sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,algorithm='auto')
# n_neighbors= 就是K-近邻里面的k值
# n_neighbors：int,可选（默认= 5），k_neighbors查询默认使用的邻居数
# algorithm一般就选auto
# algorithm：{‘auto’，‘ball_tree’，‘kd_tree’，‘brute’}，
# 可选用于计算最近邻居的算法：‘ball_tree’将会使用 BallTree，‘kd_tree’将使用 KDTree。
# ‘auto’将尝试根据传递给fit方法的值来决定最合适的算法。 (不同实现方式影响效率)
# 使用场景：小数据场景几千-几万样本

# 导入iris数据集
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def knn_iris():
    """
    用KNN算法对iris进行分类
    :return:
    """
    # 1.获取数据
    iris = load_iris()

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data,
                                                        iris.target,
                                                        test_size=0.25,
                                                        random_state=6)

    # 3.特征工程：标准化
    # 该案例不需要特征降维，因为一共就4个特征
    # 3.1实例一个转换器类
    transfer = StandardScaler()
    # 3.2调用fit_transform
    x_train = transfer.fit_transform(x_train)  # 对训练集进行标准化
    x_test = transfer.transform(x_test)        # 对测试集进行标准化

    # 4.KNN算法预估器
    estimator = KNeighborsClassifier(n_neighbors=7)
    estimator.fit(x_train, y_train)

    # 5.模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("得出预测结果y_predict:\n", y_predict)
    print("直接比对真实值和预测值：\n", y_test == y_predict)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为:\n", score)

    return None


if __name__ == '__main__':
    # 代码1:用KNN算法对iris进行分类
    knn_iris()
