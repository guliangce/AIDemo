import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


class MyDataClass:
    '''
    尝试清理数据，1查找异常点。2会的成分分析，考虑是否降低维度。
    使用knn算法进行预测，查看精确度，查看混沌矩阵
    使用KNN算法 n_neighbors 从1 到21 查看准确度。
    '''
    def __init__(self,file_o,file_n):
        # 获取数据文件
        self.data_train = pd.read_csv(file_o)
        # 获取剔除异常数据后的数据
        self.data_n = pd.read_csv(file_n)
        #
        self.x = self.data_train.drop(['y'],axis=1)
        self.y = self.data_train.loc[:,'y']
        #
        self.x_n = self.data_n.drop(['y'],axis=1)
        self.y_n = self.data_n.loc[:, 'y']
        #训练数据和测试数据分离
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x_n,self.y_n,random_state=4,test_size=0.4)
        print(self.x_train.shape,self.x_test.shape,self.x_n.shape)
        print(type(self.x_train), type(self.x_n))

    def find_exceptions(self):
        self.ee = EllipticEnvelope(contamination=0.02)
        self.ee.fit(self.x[self.y==0])
        print(self.ee.predict(self.x[self.y==0]))
        print(self.x[self.y==0])

    def dimensionality_eduction(self):
        '''
        标准化后，查看是否需要降维，查看各维度的成分比例。
        :return:
        '''
        self.x_n_n = StandardScaler().fit_transform(self.x_n)
        pca = PCA(n_components=2)
        pca.fit_transform(self.x_n_n)
        print(pca.explained_variance_ratio_)
    def create_model_1(self):
        self.nn1 = KNeighborsClassifier(n_neighbors=10)
        self.nn1.fit(self.x_n,self.y_n)
        self.nn1.predict(self.x_n)



    def draw(self):
        #绘制预测图形
        xx,yy = np.meshgrid(np.arange(0,10,0.05),np.arange(0,10,0.05))
        x_r = np.c_[xx.ravel(),yy.ravel()]

        y_r = self.nn1.predict(x_r)

        fig = plt.figure()
        bad = plt.scatter(x_r[:,0][y_r==0],x_r[:,1][y_r==0])
        good = plt.scatter(x_r[:,0][y_r==1],x_r[:,1][y_r==1])


        #绘制 n_neighbors 从1 到 21 的训练和测试准确度
        n = [i for i in range(1,21)]
        a_train = []
        a_test = []
        for i in n:
            knn = KNeighborsClassifier(n_neighbors=i)

            knn.fit(np.ascontiguousarray(self.x_train),np.ascontiguousarray(self.y_train))
            y_tr_p = knn.predict(np.ascontiguousarray(self.x_train))
            y_t_p = knn.predict(np.ascontiguousarray(self.x_test))




            a_train.append(accuracy_score(self.y_train,y_tr_p))
            a_test.append(accuracy_score(self.y_test,y_t_p))

        print(a_train)
        print(a_test)

        fig = plt.subplot(121)
        plt.plot(n,a_train)
        fig = plt.subplot(122)
        plt.plot(n,a_test)

        plt.show()

        print(x_r)

    def obtain_accuracy(self):

        print(accuracy_score(self.y_n,self.nn1.predict(self.x_n)))
        print(accuracy_score(self.y_test,self.nn1.predict(self.x_test)))
        cm = confusion_matrix(self.y_test,self.nn1.predict(self.x_test))
        print(cm)

        tp = cm[1,1]
        tn = cm[0,0]
        fp = cm[0,1]
        fn = cm[1,0]

        accuracy =  (tp + tn)/(tp + tn + fp + fn)
        #灵敏度 正样本 预期正确的比例
        recall = tp/(tp + fn)
        #特异度 负样本中 预测正确的比例
        specificity = tn/(tn+fp)
        #精确度 预测为正的样本中 预测正确的比例
        precision = tp/(tp+fp)
        #F1分数
        f1 = 2*precision*recall/(precision+recall)

        print(accuracy,recall,specificity,precision,f1)





if __name__ == '__main__':
    mdc = MyDataClass('data_class_raw.csv','data_class_processed.csv')
    mdc.find_exceptions()
    mdc.dimensionality_eduction()
    mdc.create_model_1()
    mdc.obtain_accuracy()
    mdc.draw()
