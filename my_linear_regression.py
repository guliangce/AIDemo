import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def my_print(*args):
    # 用于调试，不使用直接pass
    # pass
    print(*args)

class MyLinearRegression:
    def __init__(self,file):

        #获取数据文件
        self.data = pd.read_csv(file)
        #获取模型因子
        self.x = None
        #获取训练数据
        self.y = None
        #创建模型
        self.lr_model = LinearRegression()

    def get_x(self,list,ind1=None,ind2=None,):
        '''

        :param ind1:
        :param ind2:
        :param list:
        :return:
        '''

        #获取 模型因子
        self.x = self.data.loc[ind1:ind2, list]

    def get_y(self,list,ind1=None,ind2=None):
        # 获取训练数据
        self.y = self.data.loc[ind1:ind2, list]

    def get_predict_y(self):
        return self.lr_model.predict(self.x)

    def create_model(self):
        self.lr_model.fit(self.x, self.y)

    def check_model(self,accuracy=0.9):
        y_p = self.lr_model.predict(self.x)
        MSE = mean_squared_error(self.y, y_p)
        R2 = r2_score(self.y, y_p)
        #判断模型准确率
        if R2 < accuracy:
            print("模型训练 不 合格！")
        else:
            print("模型训练 合格！")

    def draw(self,title=None,xlabel=None,ylabel=None):
        a=plt.figure()
        sp1 = a.add_subplot(3,1,1)
        sp2 = a.add_subplot(3,1,2)
        #plt.xlabel('x')
        #plt.ylabel('y')
        sp1.scatter(self.x, self.y)
        sp2.plot(self.x, self.lr_model.predict(self.x),'r')
        plt.show()
