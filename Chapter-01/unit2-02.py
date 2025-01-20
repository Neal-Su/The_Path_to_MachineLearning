# 模型选择与调优,以对KNN算法对iris分类预测案例增加k值调优为例

# 交叉验证
# 交叉验证目的：为了让被评估的模型更加准确可信
# 什么是交叉验证：交叉验证：将拿到的训练数据，分为训练集和验证集。
# 将数据分成若干份，其中一份作为验证集。然后经过若干次(组)的测试，每次都更换不同的验证集。
# 即得到若干组模型的结果，取平均值作为最终结果。又称若干折交叉验证。
# 训练集：训练集+验证集  这行就是交叉验证
# 测试集：测试集

# 网格搜索
# 超参数搜索-网格搜索
# 通常情况下，有很多参数是需要手动指定的（如k-近邻算法中的K值），这种叫超参数。
# 但是手动过程繁杂，所以需要对模型预设几种超参数组合。
# 每组超参数都采用交叉验证来进行评估。最后选出最优参数组合建立模型。

# sklearn.model_selection.GridSearchCV(estimator, param_grid=None,cv=None)
# 对估计器的指定参数值进行详尽搜索
# estimator：估计器对象
# param_grid的含义就是想要尝试穷举的超参数数值以字典的形式传进来，如[1, 3, 5, 7, 9, 11]
# 就我想要穷举这几个k值并找出其中最优的k值，2,4这样未被传入的值不会得出
# param_grid：估计器参数(dict){“n_neighbors”:[1,3,5......]}
# cv：指定几折交叉验证
# fit：输入训练数据
# score：准确率
# 结果分析：
# best_params_: 最佳参数
# best_score_: 在交叉验证中验证的最好结果(最佳结果)
# best_estimator_: 最好的参数模型(最佳估计器)
# cv_results_: 每次交叉验证后的验证集准确率结果和训练集准确率结果(交叉验证结果)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def knn_iris_gscv():
    """
    用KNN算法对iris进行分类,添加网格搜索和交叉验证
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
    # 3.1实例一个转换器类
    transfer = StandardScaler()
    # 3.2调用fit_transform
    x_train = transfer.fit_transform(x_train)  # 对训练集进行标准化
    x_test = transfer.transform(x_test)        # 对测试集进行标准化

    # 4.KNN算法预估器
    # 因为要去试验k值哪个比较好，所以这里就不用加参数n_neighbors=了
    estimator = KNeighborsClassifier()

    # 加入网格搜索与交叉验证
    # 参数准备
    param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)

    estimator.fit(x_train, y_train)

    # 5.模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值：\n", y_test == y_predict)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为:\n", score)

    # best_params_: 最佳参数
    print("最佳参数:\n", estimator.best_params_)
    # best_score_: 在交叉验证中验证的最好结果(最佳结果)
    print("最佳结果:\n", estimator.best_score_)
    # best_estimator_: 最好的参数模型(最佳估计器)
    print("最佳估计器:\n", estimator.best_estimator_)
    # cv_results_: 每次交叉验证后的验证集准确率结果和训练集准确率结果(交叉验证结果)
    print("交叉验证结果:\n", estimator.cv_results_)

    return None


if __name__ == '__main__':
    # 代码1:用KNN算法对iris进行分类,添加网格搜索和交叉验证
    knn_iris_gscv()
