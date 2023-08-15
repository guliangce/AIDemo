import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from  sklearn.cluster import KMeans
from  sklearn.metrics import accuracy_score
from sklearn.cluster import MeanShift,estimate_bandwidth

def my_print(*args):
    # 用于调试，不使用直接pass
    # pass
    print(*args)

class MyKmeans:

    '''
    kmeans算法，一种非监督算法，
    1.根据给定的k值，选取k个样本点作为初始划分中心；
    2.计算所有样本点到每一个划分中心的距离，并将所有样本点划分到距离最近的划分中心；
    3.计算每个划分中样本点的平均值，将其作为新的中心；
    循环进行2~3步直至达到最大迭代次数，或划分中心的变化小于某一预定义阈值
    '''

    def __init__(self,file):
        # 获取数据文件
        self.data = pd.read_csv(file)

        #获取训练数据
        self.x = self.data.loc[:,['X 1','X 2']]
        self.y = self.data.loc[:,['category']]

        #kmean 模型初始化
        self.km = KMeans(n_clusters=3,random_state=0)

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
        cy = self.rectification()
        ncy = np.array(cy)
        class_py0 = plt.scatter(self.data.loc[:, 'X 1'][ncy==0], self.data.loc[:, 'X 2'][ncy==0])
        class_py1 = plt.scatter(self.data.loc[:, 'X 1'][ncy==1], self.data.loc[:, 'X 2'][ncy==1])
        class_py2 = plt.scatter(self.data.loc[:, 'X 1'][ncy==2], self.data.loc[:, 'X 2'][ncy==2])

        plt.show()

    def create_model_kmeans(self):
        self.km.fit(self.x)

    def get_predicted_data(self):
        return self.km.predict(self.x)

    def get_accuracy(self):
        my_print(pd.value_counts(self.get_predicted_data()))
        #转换Y为0维数组
        #my_print(pd.value_counts(np.array(self.y).reshape(-1)))
        my_print(pd.value_counts(self.rectification()))
        return accuracy_score(self.y,self.rectification())

    def rectification(self):
        py = self.get_predicted_data()
        cy = []
        for i in py:
            if i == 0:
                cy.append(1)
            elif i == 1:
                cy.append(2)
            elif i == 2:
                cy.append(0)
        return cy

    def predict_single(self,x1,x2):
        return self.km.predict([[x1,x2]])


if __name__ == '__main__':
    mul = MyKmeans('my_unsupervised_learning.csv')
    mul.create_model_kmeans()
    print(mul.get_accuracy())
    print(mul.predict_single(80,10))
    mul.draw()
