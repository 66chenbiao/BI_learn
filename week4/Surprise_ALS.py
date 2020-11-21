# -*- encoding: utf-8 -*-
"""
@File    : Surprise_ALS.py
@Time    : 2020/11/21 14:33
@Author  : biao chen
@Email   : 1259319710@qq.com
@Software: PyCharm
"""
from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly, KNNBasic
from surprise import accuracy
from surprise.model_selection import KFold



# 数据读取
file_path = 'E:/python/machina/kaggle_practice/week4/data/ratings.csv'
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)
train_set = data.build_full_trainset()


'''
    ALS参数:
        reg_i：物品的正则化参数，默认为10。
        reg_u：用户的正则化参数，默认为15 。
        n_epochs：迭代次数，默认为10
'''
# Baseline算法，使用ALS进行优化
bsl_options = {'method': 'als','n_epochs': 5,'reg_u': 12,'reg_i': 5}
algo = BaselineOnly(bsl_options=bsl_options)
# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    accuracy.rmse(predictions, verbose=True)
uid = str(196)
iid = str(302)
pred = algo.predict(uid, iid, r_ui=4, verbose=True)
