# 导入需要用到的算法模块
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits


# 加载数据
digits = load_digits()
data = digits.data

# 将25%的数据作为测试集，其余作为训练集
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)

# 数据预处理采用Z-Score规范化
pre_ss = preprocessing.StandardScaler()
train_pre_ss_x = pre_ss.fit_transform(train_x)
test_pre_ss_x = pre_ss.transform(test_x)

# 训练DecisionTree分类器
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0,splitter='best',criterion='gini') # sklearn默认使用基尼Gini系数
clf.fit(train_x,train_y)

predict_y = clf.predict(test_pre_ss_x)
# CART算法准确率:10.89%
print('CART算法准确率:{:.2%}'.format(accuracy_score(test_y, predict_y)))