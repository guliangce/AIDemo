import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import MeanShift,estimate_bandwidth

def my_print(*args):
    # 用于调试，不使用直接pass
    # pass
    print(*args)

class MyKnn:

    '''
    knn算法，一种监督算法，
    1.给出训练数据，和数据的分类，创建模型。
    2.给出一个待分类的数据，计算以该数据为圆心，半径内的样板那一类样板占多数？
    3.设定待分类数据的类型为多数样本的类型。
    循环进行2~3步直至达到最大迭代次数，或划分中心的变化小于某一预定义阈值
    '''

    def __init__(self,file):
        # 获取数据文件
        self.data = pd.read_csv(file)

        #获取训练数据
        self.x = self.data.loc[:,['X 1','X 2']]
        self.y = self.data.loc[:,['category']]

        #knn 模型初始化
        self.knn = KNeighborsClassifier(n_neighbors=3)

    def draw(self):

        figy = plt.subplot(121)
        mask0 = self.data.loc[:, 'category'] == 0
        mask1 = self.data.loc[:, 'category'] == 1
        mask2 = self.data.loc[:, 'category'] == 2

        class0 = plt.scatter(self.data.loc[:, 'X 1'][mask0], self.data.loc[:, 'X 2'][mask0])
        class1 = plt.scatter(self.data.loc[:, 'X 1'][mask1], self.data.loc[:, 'X 2'][mask1])
        class2 = plt.scatter(self.data.loc[:, 'X 1'][mask2], self.data.loc[:, 'X 2'][mask2])
        plt.show()

    def create_model_knn(self):
        self.knn.fit(self.x,self.y)

    def get_predicted_data(self):

        return self.knn.predict(self.x)
    def get_accuracy(self):

        return accuracy_score(self.y,self.get_predicted_data())

    def predict_single(self,x1,x2):
        return self.knn.predict([[x1,x2]])

if __name__ == '__main__':
    mknn = MyKnn('my_unsupervised_learning.csv')
    mknn.create_model_knn()
    print(mknn.predict_single(204,211))
    print(mknn.get_accuracy())

