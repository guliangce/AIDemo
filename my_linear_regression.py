import pandas as pd
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

        :param ind1:列的列表
        :param ind2:行的上标
        :param list:行的下标
        :return:
        '''

        #获取 模型因子
        self.x = self.data.loc[ind1:ind2, list]

    def get_y(self,list,ind1=None,ind2=None):
        '''
        :param list: 列的列表
        :param ind1: 行的上标
        :param ind2: 行的下标
        :return:
        '''
        # 获取训练数据
        self.y = self.data.loc[ind1:ind2, list]

    def get_predict_y(self):
        '''
        :return:获得模型预测的y
        '''
        return self.lr_model.predict(self.x)

    def create_model(self):
        '''
        创建模型训练
        :return:
        '''
        self.lr_model.fit(self.x, self.y)

    def check_model(self,accuracy=0.9):
        '''
        检查模型准确率 通过 均方误差 MSE 和R方值R2来判断。
        MSE 越小越很好，R2 越接近1越好
        :param accuracy:可设置的准确率
        :return:
        '''
        y_p = self.lr_model.predict(self.x)
        MSE = mean_squared_error(self.y, y_p)
        R2 = r2_score(self.y, y_p)
        #判断模型准确率
        if R2 < accuracy:
            print("模型训练 不 合格！")
        else:
            print("模型训练 合格！")
