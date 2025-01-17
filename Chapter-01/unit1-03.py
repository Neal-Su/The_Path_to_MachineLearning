# 英文文本特征提取CountVectorizer

# 导入iris数据集
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# CountVectorizer：统计每个样本特征值出现的个数


def count_demo():
    """
    文本特征提取CountVectorizer
    :return:
    """
    data = ["life is short,i like like python",
            "life is too long,i dislike python"]
    # 1、实例化一个转换器类
    # transfer = CountVectorizer(sparse=False),CountVectorizer中没有内置sparse=这个参数
    # 需要再结果输出的地方使用.toarray方法将sparse关闭
    # stop_words=   是用来设置停用词的，删去那些对处理没有什么用的词
    transfer = CountVectorizer(stop_words=["is", "too"])
    # 2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new：\n", data_new.toarray())  # 这里使用.toarray方法将sparse关闭
    print("返回特征名字：\n", transfer.get_feature_names_out())
    return None


if __name__ == '__main__':
    # 代码3：文本特征提取
    count_demo()

