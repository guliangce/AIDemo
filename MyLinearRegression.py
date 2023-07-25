import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def my_print(*args):
    # 用于调试，不使用直接pass
    # pass
    print(*args)


class MyLinearRegression:
    def __init__(self):
        self.data = pd.read_csv('lr_generated_data.csv')
        self.x = self.data.loc[:, ['x']]
        self.y = self.data.loc[:, ['y']]
        self.lr_model = LinearRegression()

    def create_model(self):
        self.lr_model.fit(self.x, self.y)

    def check_model(self):
        y_p = self.lr_model.predict(self.x)
        MSE = mean_squared_error(self.y, y_p)
        R2 = r2_score(self.y, y_p)
        print(MSE, R2)

    def draw(self):
        plt.figure()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(self.x, self.y)
        plt.plot(self.x, self.lr_model.predict(self.x))
        plt.show()
