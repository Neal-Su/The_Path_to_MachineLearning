# 中文文本特征提取(Tf-idf)

from sklearn.feature_extraction.text import CountVectorizer
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF的主要思想是：如果某个词或短语在一篇文章中出现的概率高，并且在其他文章中很少出现
# 则认为此词或者短语具有很好的类别区分能力，适合用来分类
# TF-IDF作用：用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度


def cut_word(text):
    """
    进行中文分词："我爱北京天安门"————>"我 爱 北京 天安门"
    :param text:
    :return:
    """
    return " ".join(list(jieba.cut(text)))


def tfidf_demo():
    """
    中文文本特征提取(Tf-idf)
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
    transfer = TfidfVectorizer(stop_words=["一种"])
    # 2、调用fit_transform
    data_new2 = transfer.fit_transform(text_list)
    print("data_new2：\n", data_new2.toarray())  # 这里使用.toarray方法将sparse关闭
    print("返回特征名字：\n", transfer.get_feature_names_out())


if __name__ == '__main__':
    # 代码6：中文文本特征提取(Tf-idf)
    tfidf_demo()