# -*- encoding: utf-8 -*-
"""
@File    : Aprior_home.py
@Time    : 2020/11/7 23:37
@Author  : biao chen
@Email   : 1259319710@qq.com
@Software: PyCharm
"""

# 导入所需的模块
import time
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# 转换函数
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


# 数据加载
data = pd.read_csv('E:/python/machina/kaggle_practice/data/BreadBasket_DMS.csv')
# 统一小写
data['Item'] = data['Item'].str.lower()
# 去掉none项
data = data.drop(data[data.Item == 'none'].index)

# 设置最大的列数
pd.options.display.max_columns=100
# 开始时间
start = time.time()
# 数据处理
hot_encoded_df=data.groupby(['Transaction','Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')
hot_encoded_df = hot_encoded_df.applymap(encode_units)
# 调用apriori方法
frequent_itemsets = apriori(hot_encoded_df, min_support=0.03, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.4)
print("频繁项集：", frequent_itemsets)
# 输出满足 rules['lift'] >=1 并且 rules['confidence'] >=0.5 的关联规则
print("关联规则：", rules[ (rules['lift'] >= 1) & (rules['confidence'] >= 0.5) ])
# 结束时间
end = time.time()
print("用时：", end-start)
