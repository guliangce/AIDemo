import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class MyPca:
    '''
    降维处理模块，在维度比较多的场景下，降维更方便处理和可视化。
    1、标准化数据。
    2、同等维度降维，观察方差，方差越小的维度可删除。
    3、保留占比高的维度。
    4、再次降维。
    5、对比降维后的预测率.
    '''
    def __init__(self,file):
        # 获取数据文件
        self.data = pd.read_csv(file)

        # 获取训练数据
        self.x = self.data.loc[:, ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
        self.y = self.data.loc[:, 'category']

        #knn 模型初始化
        self.knn = KNeighborsClassifier(n_neighbors=3)

        #标准化方差的均值后的X
        self.nrom_x = StandardScaler().fit_transform(self.x)

    def original_knn(self):
        self.knn.fit(self.x,self.y)

    def new_knn(self,x):
        self.knn.fit(x,self.y)

    def get_predict(self,x):
        return self.knn.predict(x)

    def get_original_accuracy(self):
        return accuracy_score(self.y, self.knn.predict(self.x))

    def get_new_accuracy(self,n_y):
        return accuracy_score(self.y, n_y)

    def dimensionality_reduction(self,Dimension):
        pca = PCA(n_components=Dimension)
        dr = pca.fit_transform(self.nrom_x)
        print(pca.explained_variance_ratio_)
        return dr

    def draw(self,data,y):
        fig = plt.figure()
        class0 = plt.scatter(data[:,0][y==0], data[:,1][y==0])
        class1 = plt.scatter(data[:, 0][y == 1], data[:, 1][y == 1])
        class2 = plt.scatter(data[:, 0][y == 2], data[:, 1][y == 2])
        plt.show()



if __name__ == '__main__':
    pca = MyPca('iris.csv')
    pca.original_knn()
    print(pca.get_original_accuracy())
    pca.dimensionality_reduction(4)
    dr = pca.dimensionality_reduction(2)
    pca.draw(dr,pca.y)
    pca.new_knn(dr)
    n_y = pca.get_predict(dr)
    pca.draw(dr,n_y)
    print(pca.get_new_accuracy(n_y))

