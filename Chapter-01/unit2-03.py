# 预测Facebook签到位置案例

# 数据介绍：将根据用户的位置，准确性和时间戳预测用户正在查看的业务。
# row_id：登记事件的ID
# xy：坐标
# 准确性：定位准确性
# 时间：时间戳
# place_id：业务的ID，这是您预测的目标

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def knncls():
    """
    K近邻算法预测位置类别
    :return:
    """
    # 1.获取数据
    data_path = r"E:\B python wenjian\train.csv"
    data = pd.read_csv(data_path)

    # 2.基本的数据处理
    # 2.1缩小数据范围
    data = data.query("x < 2.5 & x > 2 & y < 1.5 & y > 1.0")
    # 2.2删除time这一列特征
    data = data.drop(['time'], axis=1)
    # 2.3删除入住次数少于三次位置
    place_count = data.groupby('place_id').count()
    tf = place_count[place_count.row_id > 3].reset_index()
    data = data[data['place_id'].isin(tf.place_id)]

    # 3、取出特征值和目标值
    y = data['place_id']
    x = data.drop(['place_id', 'row_id'], axis=1)

    # 4.数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # 5.特征工程：标准化
    # 5.1实例一个转换器类
    transfer = StandardScaler()
    # 5.2调用fit_transform
    x_train = transfer.fit_transform(x_train)  # 对训练集进行标准化
    x_test = transfer.transform(x_test)  # 对测试集进行标准化

    # 6.KNN算法预估器
    # 因为要去试验k值哪个比较好，所以这里就不用加参数n_neighbors=了
    estimator = KNeighborsClassifier()

    # 加入网格搜索与交叉验证
    # 参数准备
    param_dict = {"n_neighbors": [3, 5, 7, 9]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)

    estimator.fit(x_train, y_train)

    # 7.模型评估
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


if __name__ == '__main__':
    # 代码4:用朴素贝叶斯算法对新闻进行分类
    knncls()
