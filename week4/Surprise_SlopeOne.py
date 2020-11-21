# -*- encoding: utf-8 -*-
"""
@File    : Surprise_SlopeOne.py
@Time    : 2020/11/21 14:59
@Author  : biao chen
@Email   : 1259319710@qq.com
@Software: PyCharm
"""
# 博客地址： https://blog.csdn.net/xidianliutingting/article/details/51916578
from surprise import Dataset
from surprise import Reader
from surprise import SlopeOne
from surprise import accuracy
from surprise.model_selection import KFold

# 数据读取
file_path = 'E:/python/machina/kaggle_practice/week4/data/ratings.csv'
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)
train_set = data.build_full_trainset()

'''
    该算法适用于物品更新不频繁，数量相对较稳定并且物品数目明显小于用户数的场景。依赖用户的用户行为日志和物品偏好的相关内容。
    优点：
    1.算法简单，易于实现，执行效率高；
    2.可以发现用户潜在的兴趣爱好；
    缺点：
    依赖用户行为，存在冷启动问题和稀疏性问题。

'''
# SlopeOne
algo = SlopeOne()
# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    accuracy.rmse(predictions, verbose=True)
uid = str(196)
iid = str(302)
pred = algo.predict(uid, iid, r_ui=4, verbose=True)
print(pred)









