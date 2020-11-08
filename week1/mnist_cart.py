# ������Ҫ�õ����㷨ģ��
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits


# ��������
digits = load_digits()
data = digits.data

# ��25%��������Ϊ���Լ���������Ϊѵ����
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)

# ����Ԥ�������Z-Score�淶��
pre_ss = preprocessing.StandardScaler()
train_pre_ss_x = pre_ss.fit_transform(train_x)
test_pre_ss_x = pre_ss.transform(test_x)

# ѵ��DecisionTree������
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0,splitter='best',criterion='gini') # sklearnĬ��ʹ�û���Giniϵ��
clf.fit(train_x,train_y)

predict_y = clf.predict(test_pre_ss_x)
# CART�㷨׼ȷ��:10.89%
print('CART�㷨׼ȷ��:{:.2%}'.format(accuracy_score(test_y, predict_y)))