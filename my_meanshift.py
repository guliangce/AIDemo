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

class MyMeanShift:

    '''
    meanshift算法，一种非监督算法，
    1.通过样本和给定的计算样本数，计算出 算法半径。
    2.建立模型，以计算的算法半径为基础，取半径内样本，计算随机点和样本的距离，划分分类。
    3.获取所有已分类的样本的分类中心，进行偏移，再次计算分类。直到分类固定不变。

    '''

    def __init__(self,file):
        # 获取数据文件
        self.data = pd.read_csv(file)

        #获取训练数据
        self.x = self.data.loc[:,['X 1','X 2']]
        self.y = self.data.loc[:,['category']]

        #获取训练半径
        self.bw = estimate_bandwidth(self.x,n_samples=10)
        my_print(self.bw)
        #knn 模型初始化
        self.kms = MeanShift(bandwidth=self.bw)

    def draw(self):

        figy = plt.subplot(121)
        mask0 = self.data.loc[:, 'category'] == 0
        mask1 = self.data.loc[:, 'category'] == 1
        mask2 = self.data.loc[:, 'category'] == 2

        class0 = plt.scatter(self.data.loc[:, 'X 1'][mask0], self.data.loc[:, 'X 2'][mask0])
        class1 = plt.scatter(self.data.loc[:, 'X 1'][mask1], self.data.loc[:, 'X 2'][mask1])
        class2 = plt.scatter(self.data.loc[:, 'X 1'][mask2], self.data.loc[:, 'X 2'][mask2])

        fig_py = plt.subplot(122)
        py = self.get_predicted_data()
        cy = self.rectification(self.get_predicted_data())
        ncy = np.array(cy)
        class_py0 = plt.scatter(self.data.loc[:, 'X 1'][ncy == 0], self.data.loc[:, 'X 2'][ncy == 0])
        class_py1 = plt.scatter(self.data.loc[:, 'X 1'][ncy == 1], self.data.loc[:, 'X 2'][ncy == 1])
        class_py2 = plt.scatter(self.data.loc[:, 'X 1'][ncy == 2], self.data.loc[:, 'X 2'][ncy == 2])

        plt.show()

    def create_model_kms(self):
        self.kms.fit(self.x)

    def get_predicted_data(self):

        return self.kms.predict(self.x)

    def rectification(self,py):
        cy = []
        for i in py:
            if i == 0:
                cy.append(2)
            elif i == 1:
                cy.append(1)
            elif i == 2:
                cy.append(0)
        return cy

    def get_accuracy(self):
        #转换Y为0维数组
        my_print(pd.value_counts(np.array(self.y).reshape(-1)))
        my_print(pd.value_counts(self.get_predicted_data()))
        my_print(pd.value_counts(self.rectification(self.get_predicted_data())))

        return accuracy_score(self.y,self.rectification(self.get_predicted_data()))

    def predict_single(self,x1,x2):
        return self.kms.predict([[x1,x2]])

if __name__ == '__main__':
    mms = MyMeanShift('my_unsupervised_learning.csv')
    mms.create_model_kms()
    print(mms.rectification(mms.predict_single(204,211)))
    print(mms.get_accuracy())
    mms.draw()

