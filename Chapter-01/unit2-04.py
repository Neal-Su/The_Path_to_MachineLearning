# 朴素贝叶斯算法,以对20类新闻进行分类为例

# 朴素的意思是：假设特征与特征之间是相互独立的
# 应用场景：文本分类，单词作为特征

# 拉普拉斯平滑系数：
# 目的：防止计算出的分类概率为0

# sklearn.naive_bayes.MultinomialNB(alpha = 1.0)
# 朴素贝叶斯分类
# alpha：拉普拉斯平滑系数

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def nb_news():
    """
    用朴素贝叶斯算法对新闻进行分类
    :return:
    """
    # 1.获取数据
    # subset: {'train', 'test', 'all'}, default = 'train'
    # Select the dataset to load: 'train' for the training set,
    # 'test' for the test set, 'all' for both, with shuffled ordering.
    news = fetch_20newsgroups(subset="all")

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(news.data,news.target)

    # 3.特征工程,对文本进行特征抽取(tf-idf方法)
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)  # 对训练集进行标准化
    x_test = transfer.transform(x_test)        # 对测试集进行标准化

    # 4.朴素贝叶斯算法预估器
    # alpha：拉普拉斯平滑系数默认为1
    estimator = MultinomialNB()
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
    # 代码4:用朴素贝叶斯算法对新闻进行分类
    nb_news()
