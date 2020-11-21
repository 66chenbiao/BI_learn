# -*- encoding: utf-8 -*-
"""
@File    : Surprise_SGD.py
@Time    : 2020/11/21 14:41
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
    SGD参数:
        reg：代价函数的正则化项，默认为0.02。
        learning_rate：学习率，默认为0.005。
        n_epochs：迭代次数，默认为20。

'''
# Baseline算法，使用SGD进行优化
bsl_options = {'method': 'sgd','n_epochs': 5}
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
print(pred)
# 迭代速度比ALS快