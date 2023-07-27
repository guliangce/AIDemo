import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def my_print(*args):
    # 用于调试，不使用直接pass
    # pass
    print(*args)

class MyLogicRegression:
    def __init__(self,file):

        #获取数据文件
        self.data = pd.read_csv(file)
        #获取训练数据
        self.x = self.data.loc[:,['Exam 1','Exam 2']]
        self.y = self.data.loc[:,['pass']]
        #创建模型
        #边界函数：
        #th=theta
        #一阶 th0+th1*x1+th2*x2 = 0
        #二阶 th0+ht1*x1+th2*x2+th3*x1**+th4*x2**+th5*x1*x2 = 0
        self.loc_model = LogisticRegression()
        #创建二阶因子
        self.x1 = self.data.loc[:,'Exam 1']
        self.x2 = self.data.loc[:,'Exam 2']
        self.x1_x1 = self.x1 * self.x1
        self.x2_x2 = self.x2 * self.x2
        self.x1_x2 = self.x1 * self.x2
        #格式化二阶的训练参数
        self.x_second_order = pd.DataFrame({'x1':self.x1,'x2':self.x2,'x1_x1':self.x1_x1,'x2_x2':self.x2_x2,'x1_x2':self.x1_x2})


    def draw(self):
        fig = plt.figure()
        mask = self.data.loc[:, 'pass']==1
        p = plt.scatter(self.data.loc[:,'Exam 1'][mask],self.data.loc[:,'Exam 2'][mask])
        f = plt.scatter(self.data.loc[:, 'Exam 1'][~mask], self.data.loc[:, 'Exam 2'][~mask])
        plt.legend((p,f),('pass','failed'))
        my_print(self.loc_model.coef_)
        self.x1 = self.x1.sort_values()

        if len(self.loc_model.coef_[0]) == 5:
            #边界曲线公式
            #step1:th0+ht1*x1+th2*x2+th3*x1**+th4*x2**+th5*x1*x2 = 0
            #step2:th4*x2** + (th5x1 + th2)*x2 + (th0 + ht1*x1 + th3*x1*x1) = 0
            #step3:a*x*x + b*x + c = 0
            #step4:
            th0 = self.loc_model.intercept_
            th1,th2,th3,th4,th5 = self.loc_model.coef_[0][0],self.loc_model.coef_[0][1], \
                                    self.loc_model.coef_[0][2],self.loc_model.coef_[0][3], \
                                    self.loc_model.coef_[0][4]
            #a
            a = th4
            b = th5 * self.x1 + th2
            c = th0 + th1*self.x1 + th3*self.x1*self.x1
            x2 = (-b + np.sqrt(b*b-4*a*c))/(2*a)
            plt.plot(self.x1,x2)
        plt.show()

    def create_model(self,X):
        #X 可以选择一阶数据或二阶
        self.loc_model.fit(X,self.y)

    def get_predicted_data(self,x):
        #获取训练好的模型对训练数据X的预测值
        return self.loc_model.predict(x)

    def get_accuracy(self,x):
        #获得准确率
        return accuracy_score(self.y,self.get_predicted_data(x))

    def get_single_predicted_1(self,*args):
        #获取单个的通过情况。
        if self.loc_model.predict([args]) == 1:
            return '通过考试！'
        else:
            return '不通过！'

    def get_single_predicted_2(self,*args):
        list = []
        list.append(args[0])
        list.append(args[1])
        list.append(args[0] * args[0])
        list.append(args[1] * args[1])
        list.append(args[0] * args[1])
        my_print(list)


        #获取单个的通过情况。
        if self.loc_model.predict([list]) == 1:
            return '通过考试！'
        else:
            return '不通过！'


