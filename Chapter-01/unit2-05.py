# 决策树算法,对iris分类为例

# 决策树的划分依据之一：信息增益

# class sklearn.tree.DecisionTreeClassifier(criterion="gini", max_depth=None,random_state=None)
# 决策树分类器
# criterion:默认是’gini’系数，也可以选择信息增益的熵"entropy"
# max_depth:树的深度大小
# random_state:随机数种子
# 其中会有些超参数：max_depth:树的深度大小

# 决策树可视化:
# from sklearn.tree import export_graphviz
# sklearn.tree.export_graphviz() 该函数能够导出DOT格式
# tree.export_graphviz(estimator,out_file='tree.dot’,feature_names=[‘’,’’])
# 决策树dot文件可视化网站：https://dreampuf.github.io/GraphvizOnline

# 决策树小结:
# 缺点：
# 决策树学习者可以创建不能很好地推广数据的过于复杂的树，这被称为过拟合。
# 改进：
# 减枝cart算法(决策树API当中已经实现，随机森林参数调优有相关介绍)
# 随机森林

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def decision_iris():
    """
    用决策树对iris进行分类
    :return:
    """
    # 1.获取数据
    iris = load_iris()

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data,
                                                        iris.target,
                                                        test_size=0.25,
                                                        random_state=6)

    # 3.决策树算法预估器
    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train, y_train)

    # 4.模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("得出预测结果y_predict:\n", y_predict)
    print("直接比对真实值和预测值：\n", y_test == y_predict)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为:\n", score)

    # 可视化决策树
    export_graphviz(estimator, out_file="iris_tree.dot", feature_names=iris.feature_names)
    # 3.决策树算法预估器
    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train, y_train)

    # 4.模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("得出预测结果y_predict:\n", y_predict)
    print("直接比对真实值和预测值：\n", y_test == y_predict)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为:\n", score)

    # 可视化决策树
    export_graphviz(estimator, out_file="iris_tree.dot", feature_names=iris.feature_names)

    return None


if __name__ == '__main__':
    # 代码5:用决策树对iris进行分类
    decision_iris()
