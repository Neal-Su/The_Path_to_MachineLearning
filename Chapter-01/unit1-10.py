# 特征降维(特征选择-过滤式-相关系数法)

# 皮尔逊相关系数(Pearson Correlation Coefficient)
# 反映变量之间相关关系密切程度的统计指标

# 相关系数的值介于–1与+1之间，即–1≤ r ≤+1。其性质如下：
# 当r>0时，表示两变量正相关，r<0时，两变量为负相关
# 当|r|=1时，表示两变量为完全相关，当r=0时，表示两变量间无相关关系
# 当0<|r|<1时，表示两变量存在一定程度的相关。且|r|越接近1，两变量间线性关系越密切；|r|越接近于0，表示两变量的线性相关越弱
# 一般可按三级划分：|r|<0.4为低度相关；0.4≤|r|<0.7为显著性相关；0.7≤|r|<1为高度线性相关

from scipy.stats import pearsonr
import pandas as pd


def demo():
    """
    :return:
    """
    # 1.获取数据
    # data_path设置路径
    data_path = r"E:\B python wenjian\factor_returns.csv"
    # 使用pandas库读取txt文件
    data = pd.read_csv(data_path)
    data = data.iloc[:, 1:-2]  # pandas库里的一个功能，取第2列至倒数第3列保留

    # 计算某两个变量之间的相关系数
    r = pearsonr(data["pe_ratio"], data["pb_ratio"])
    print("相关系数：\n", r)


if __name__ == '__main__':
    # 代码10：皮尔逊相关系数
    demo()
