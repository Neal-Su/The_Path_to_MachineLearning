# 中文文本特征提取（jieba库自动分词）

# 导入iris数据集
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import jieba


def cut_word(text):
    """
    进行中文分词："我爱北京天安门"————>"我 爱 北京 天安门"
    :param text:
    :return:
    """
    return " ".join(list(jieba.cut(text)))


def count_chinese_demo2():
    """
    中文文本特征提取（jieba库自动分词）
    :return:
    """
    # 将中文文本进行分词
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    text_list = []
    for sent in data:
        text_list.append(cut_word(sent))
    # 1、实例化一个转换器类
    # transfer = CountVectorizer(sparse=False),CountVectorizer中没有内置sparse=这个参数
    # 需要再结果输出的地方使用.toarray方法将sparse关闭
    transfer = CountVectorizer(stop_words=["一种"])
    # 2、调用fit_transform
    data_new2 = transfer.fit_transform(text_list)
    print("data_new2：\n", data_new2.toarray())  # 这里使用.toarray方法将sparse关闭
    print("返回特征名字：\n", transfer.get_feature_names_out())


if __name__ == '__main__':
    # 代码5：中文文本特征提取（jieba库自动分词）
    count_chinese_demo2()
