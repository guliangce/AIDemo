import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
def my_print(*args):
    # 用于调试，不使用直接pass
    # pass
    print(*args)

class MyTRTrain:
    '''
    尝试比对 一阶 和 二阶的模型有效性
    '''
    def __init__(self,file_train,file_test):
        # 获取数据文件
        self.data_train = pd.read_csv(file_train)
        # 获取测试文件
        self.data_test = pd.read_csv(file_test)

        self.x_train = self.data_train.loc[:,['T']]
        self.x_test = self.data_test.loc[:,['T']]

        self.y_train = self.data_train.loc[:,['rate']]
        self.y_test = self.data_test.loc[:, ['rate']]

        #创建二阶的训练和测试数据
        self.cho = PolynomialFeatures(degree=2)
        self.x_train_2 = self.cho.fit_transform(self.x_train)
        self.x_test_2 = self.cho.transform(self.x_test)

        self.lm1 = None
        self.lm2 = None

    def create_model_1(self):
        '''
        使用一阶线性进行回归，观察re_score
        :return:
        '''
        self.lm1 = LinearRegression()
        self.lm1.fit(self.x_train,self.y_train)

        print(r2_score(self.y_train,self.lm1.predict(self.x_train)))
        print(r2_score(self.y_test, self.lm1.predict(self.x_test)))

    def create_model_2(self):
        '''
        使用二阶线性进行回归，观察re_score
        :return:
        '''
        self.lm2 = LinearRegression()
        self.lm2.fit(self.x_train_2, self.y_train)

        print(r2_score(self.y_train, self.lm2.predict(self.x_train_2)))
        print(r2_score(self.y_test, self.lm2.predict(self.x_test_2)))

    def draw(self):
        '''
        将二阶线性的图形画出来。
        :return:
        '''
        x_draw = np.linspace(40,90,300).reshape(-1,1)
        x_draw_2 = self.cho.transform(x_draw)
        y_draw_2 = self.lm2.predict(x_draw_2)
        fig = plt.figure()
        plt.plot(x_draw,y_draw_2)
        plt.show()




if __name__ == '__main__':
    mtrt = MyTRTrain('T-R-train.csv','T-R-test.csv')
    mtrt.create_model_1()
    mtrt.create_model_2()
    mtrt.draw()



