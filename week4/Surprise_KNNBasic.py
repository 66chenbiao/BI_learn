# -*- encoding: utf-8 -*-
"""
@File    : Surprise_KNNBasic.py
@Time    : 2020/11/21 15:12
@Author  : biao chen
@Email   : 1259319710@qq.com
@Software: PyCharm
"""
import os
import io
from surprise import Dataset
from surprise import Reader
from surprise import KNNBaseline
from surprise import accuracy
from surprise.model_selection import KFold

# 数据集下载地址： http://files.grouplens.org/datasets/movielens/
# 获取id到name的互相映射  步骤:2
def read_item_names():
    """
    获取电影名到电影id 和 电影id到电影名的映射
    """
    #os.path.expanduser(path) 把path中包含的"~"和"~user"转换成用户目录
    file_name = (os.path.expanduser('E:/python/machina/kaggle_practice/week4/data/ml-100k') +'/u.item')
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]
    return rid_to_name, name_to_rid

# 数据读取
file_path = 'E:/python/machina/kaggle_practice/week4/data/ratings.csv'
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)
train_set = data.build_full_trainset()


# 相似度计算，使用皮尔逊相似度计算法，使用ItemCF的相似度计算
sim_options = {'name': 'pearson_baseline', 'user_based': False}
# 使用KNNBaseline算法，一种CF算法
algo = KNNBaseline(sim_options=sim_options)
algo.fit(train_set)
#获得电影名称信息数据
rid_to_name, name_to_rid = read_item_names()

#获得Toy Story电影的电影ID
toy_story_raw_id = name_to_rid['Toy Story (1995)']
print(toy_story_raw_id)
#通过Toy Story电影的电影ID获取该电影的推荐内部id
toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)
print(toy_story_inner_id)
