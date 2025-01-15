# 字典特征提取

# 导入iris数据集
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

def dict_demo():
    """
    字典特征抽取
    :return:
    """
    # 这是一个包含字典的迭代器
    data = [{'city': '北京', 'temperature': 100},
            {'city': '上海', 'temperature': 60},
            {'city': '深圳', 'temperature': 30}]
    # 1.实例化一个转换器类
    # 默认返回的是sparse（稀疏）矩阵，设置sparse=False就不会返回稀疏矩阵了
    transfer = DictVectorizer(sparse=False)
    # 2.调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new：\n", data_new)
    # 打印特征名字
    print("特征名字：\n", transfer.get_feature_names_out())
    return None


if __name__ == '__main__':
    # 代码2：字典特征抽取
    dict_demo()