# sklearn数据集的使用，以iris为例

# 导入iris数据集
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def datasets_demo():
    """
    sklearn数据集使用
    :return:
    """
    # 获取数据集
    iris = load_iris()
    print("iris数据集：\n", iris)
    # 返回值是一个继承自字典的Bench

    print("鸢尾花的描述：\n", iris.DESCR)
    print("鸢尾花的特征值:\n", iris["data"])
    print("鸢尾花的目标值：\n", iris.target)
    print("鸢尾花特征的名字：\n", iris.feature_names)
    print("鸢尾花目标值的名字：\n", iris.target_names)

    # 2、对鸢尾花数据集进行分割
    # 训练集的特征值x_train 测试集的特征值x_test 训练集的目标值y_train 测试集的目标值y_test
    x_train, x_test, y_train, y_test = train_test_split(iris.data,
                                                        iris.target,
                                                        test_size=0.2,
                                                        random_state=22)
    # test_size=0.2 代表测试集占20%
    # random_state=22 代表使用22号随机数种子
    print("x_train:\n", x_train, x_train.shape)
    # x_train.shape：检验该值有多少行多少列(120, 4)表示120行4列
    return None

if __name__ == '__main__':
    # 代码1：sklearn数据集使用
    datasets_demo()
