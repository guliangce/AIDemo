import pandas as pd
from sklearn.covariance import EllipticEnvelope
from matplotlib import pyplot as plt
import numpy as np

class MyAnomalyDetection:
    '''
    异常检测算法
    主要是通过概率密度函数，统计低于某个阈值的离散的点，即为异常数据。
    '''
    def __init__(self,file):

        # 获取数据文件
        self.data = pd.read_csv(file)

        self.x = self.data.loc[:, ['X 1', 'X 2']]

        #creat model
        self.ad = EllipticEnvelope(contamination=0.02)

    def create_model_ad(self):

        self.ad.fit(self.x)

    def get_predicted_data(self):

        return self.ad.predict(self.x)

    def draw(self):

        figy = plt.figure()
        plt.scatter(self.data.loc[:, 'X 1'], self.data.loc[:, 'X 2'])
        pd = self.get_predicted_data()
        plt.scatter(self.data.loc[:, 'X 1'][pd==-1], self.data.loc[:, 'X 2'][pd==-1])

        plt.show()


if __name__ == '__main__':
    ad = MyAnomalyDetection('Anomaly_Detection.csv')
    ad.create_model_ad()
    print(ad.get_predicted_data())
    ad.draw()
