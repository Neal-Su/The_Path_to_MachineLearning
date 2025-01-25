# 随机森林算法,用随机森林对泰坦尼克号乘客生存预测

# 什么是集成学习方法:
# 集成学习通过建立几个模型组合的来解决单一预测问题。
# 它的工作原理是生成多个分类器/模型，各自独立地学习和作出预测。
# 这些预测最后结合成组合预测，因此优于任何一个单分类的做出预测。

# 什么是随机森林:
# 在机器学习中，随机森林是一个包含多个决策树的分类器，
# 并且其输出的类别是由个别树输出的类别的众数而定。

# 随机森林原理过程:
# 学习算法根据下列算法而建造每棵树：
# 用N来表示训练用例（样本）的个数，M表示特征数目。
# 1、训练集随机:   一次随机选出一个样本，重复N次，有可能出现重复的样本）
# 2、特征随机:     随机去选出m个特征, m <<M，建立决策树
# 3.采取bootstrap抽样

# 为什么采用BootStrap随机有放回抽样
# 为什么要随机抽样训练集？　　
# 如果不进行随机抽样，每棵树的训练集都一样，那么最终训练出的树分类结果也是完全一样的
# 为什么要有放回地抽样？
# 如果不是有放回的抽样，那么每棵树的训练样本都是不同的，
# 都是没有交集的，这样每棵树都是“有偏的”，都是绝对“片面的”（当然这样说可能不对），
# 也就是说每棵树训练出来都是有很大的差异的；而随机森林最后分类取决于多棵树（弱分类器）的投票表决。

# class sklearn.ensemble.RandomForestClassifier
# (n_estimators=10, criterion=’gini’, max_depth=None, bootstrap=True, random_state=None, min_samples_split=2)
# 随机森林分类器
# n_estimators：integer，optional（default = 10）森林里的树木数量120,200,300,500,800,1200
# criteria：string，可选（default =“gini”）分割特征的测量方法
# max_depth：integer或None，可选（默认=无）树的最大深度 5,8,15,25,30
# max_features="auto”,每个决策树的最大特征数量
# If "auto", then max_features=sqrt(n_features).
# If "sqrt", then max_features=sqrt(n_features) (same as "auto").
# If "log2", then max_features=log2(n_features).
# If None, then max_features=n_features.
# bootstrap：boolean，optional（default = True）是否在构建树时使用放回抽样
# min_samples_split:节点划分最少样本数
# min_samples_leaf:叶子节点的最小样本数
# 超参数：n_estimator, max_depth, min_samples_split,min_samples_leaf

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


def random_forest_titan():
    """
    用随机森林对泰坦尼克号乘客生存预测
    :return:
    """
    # 1.获取数据
    data_path = r"E:\B python wenjian\Titanic.csv"
    titanic = pd.read_csv(data_path)

    # 2.数据处理
    # 2.1筛选目标值和特征值
    x = titanic[["pclass", "age", "sex"]]
    y = titanic["survived"]
    # 2.2缺失值处理
    x["age"].fillna(x["age"].mean(), inplace=True)
    # 2.3转换成字典
    x = x.to_dict(orient="records")

    # 3.数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=6)

    # 4.字典特征抽取
    transfer = DictVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 5.随机森林算法预估器
    estimator = RandomForestClassifier()

    # 加入网格搜索与交叉验证
    # 参数准备
    param_dict = {"n_estimators": [120,200,300,500,800,1200], "max_depth": [5,8,15,25,30]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)

    estimator.fit(x_train, y_train)

    # 6.模型评估
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
    # 代码7:用随机森林对泰坦尼克号乘客生存预测
    random_forest_titan()
