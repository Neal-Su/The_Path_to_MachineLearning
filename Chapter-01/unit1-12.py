# instacart菜篮子，特征工程案例

# 探究用户对物品类别的喜好
# 数据如下：

# order_products__prior.csv：订单与商品信息
# 字段：order_id, product_id, add_to_cart_order, reordered

# products.csv：商品信息
# 字段：product_id, product_name, aisle_id, department_id

# orders.csv：用户的订单信息
# 字段：order_id,user_id,eval_set,order_number,….

# aisles.csv：商品所属具体物品类别
# 字段： aisle_id, aisle

# 需要将用户user_id和物品类别aisle放到一个表中-合并
# 找到user_id和aisle之间的关系-交叉表和透视表
# 特征冗余过多(特征有大量的零)-PCA降维

# 这里是在jupyter上检查表格完成

import pandas as pd
from sklearn.decomposition import PCA

# 1.获取数据
aisles_path = r"E:\B python wenjian\instacart\aisles.csv"
order_products_path = r"E:\B python wenjian\instacart\order_products__prior.csv"
orders_path = r"E:\B python wenjian\instacart\orders.csv"
products_path = r"E:\B python wenjian\instacart\products.csv"
aisles = pd.read_csv(aisles_path)
order_products = pd.read_csv(order_products_path)
orders = pd.read_csv(orders_path)
products = pd.read_csv(products_path)

# 2.表的合并
# 合并aisles表和products表,按aisle_id进行合并
tab1 = pd.merge(aisles, products, on=["aisle_id", "aisle_id"])
tab2 = pd.merge(tab1, order_products, on=["product_id", "product_id"])
tab3 = pd.merge(tab2, orders, on=["order_id", "order_id"])

# 3.找到user_id和aisle之间的关系
table = pd.crosstab(tab3["user_id"], tab3["aisle"])

# 减少数据量，要不电脑跑不动
data = table[:10000]

# 4.PCA降维


def pca_demo():
    """
    PCA降维
    :return:
    """
    # 1.实例一个转换器类
    # sklearn.decomposition.PCA(n_components=None)
    # 将数据分解为较低维数空间
    # n_components:
    # 小数：表示保留百分之多少的信息
    # 整数：减少到多少个特征
    transfer = PCA(n_components=0.95)

    # 2.调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    print("data_new形状：\n", data_new.shape)

    return None


if __name__ == '__main__':
    pca_demo()
