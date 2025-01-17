# 中文文本特征提取CountVectorizer

# 导入iris数据集
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# CountVectorizer：统计每个样本特征值出现的个数


def count_chinese_demo():
    """
    文本特征提取CountVectorizer
    :return:
    """
    data = ["大连海事大学计算机科学与技术",
            "大连海事大学帆船队"]
    transfer = CountVectorizer()
    data_new = transfer.fit_transform(data)
    print("data_new：\n", data_new.toarray())  # 这里使用.toarray方法将sparse关闭
    print("返回特征名字：\n", transfer.get_feature_names_out())

    data2 = ["大连 海事 大学 计算机 科学 与 技术", "大连 海事 大学 帆船队"]
    # 1、实例化一个转换器类
    # transfer = CountVectorizer(sparse=False),CountVectorizer中没有内置sparse=这个参数
    # 需要再结果输出的地方使用.toarray方法将sparse关闭
    transfer = CountVectorizer()
    # 2、调用fit_transform
    data_new2 = transfer.fit_transform(data2)
    print("data_new2：\n", data_new2.toarray())  # 这里使用.toarray方法将sparse关闭
    print("返回特征名字：\n", transfer.get_feature_names_out())
    return None


if __name__ == '__main__':
    # 代码4：中文文本特征提取
    count_chinese_demo()