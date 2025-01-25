# 决策树算法,泰坦尼克号乘客生存预测案例

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def decision_titan():
    """
    用决策树对泰坦尼克号乘客生存预测
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

    # 5.决策树算法预估器
    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train, y_train)

    # 6.模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("得出预测结果y_predict:\n", y_predict)
    print("直接比对真实值和预测值：\n", y_test == y_predict)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为:\n", score)

    # 可视化决策树
    export_graphviz(estimator, out_file="titanic_tree.dot", feature_names=transfer.get_feature_names_out())
    return None
    # 决策树dot文件可视化网站：https://dreampuf.github.io/GraphvizOnline


if __name__ == '__main__':
    # 代码6:用决策树对泰坦尼克号乘客生存预测
    decision_titan()
