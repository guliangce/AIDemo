import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN
import tushare as ts

class MyRnn:
    def __init__(self):
        self.data = pd.read_csv("zgpa_train.csv")
        self.data.head()

        self.price = self.data.loc[:, 'close']
        self.price.head()

        self.price_norm = self.price / max(self.price)

        self.time_step = 8
        self.X, self.y = self.extract_data(self.price_norm, self.time_step)

    def darw(self):
        # fig1 = plt.figure(figsize=(10, 5))
        # plt.plot(self.price)
        # plt.title("close price")
        # plt.xlabel("time")
        # plt.ylabel("price")
        # plt.show()

        y_train_predict = self.model.predict(self.X) * max(self.price)
        y_train = [i * max(self.price) for i in self.y]

        fig2 = plt.figure(figsize=(10, 5))
        plt.plot(y_train, label="real price")
        plt.plot(y_train_predict, label="predict price")
        plt.title("price")
        plt.xlabel("time")
        plt.ylabel("price")
        plt.legend()
        plt.show()



    def extract_data(self,data, time_step):
        X = []
        y = []
        # 0,1,2,3....9 :10个样本 time_step=8; 0-7,1-8,2-9 三组
        for i in range(len(data) - time_step):
            X.append([a for a in data[i: i + time_step]])
            y.append(data[i + time_step])
        X = np.array(X)
        X = X.reshape(X.shape[0], X.shape[1], 1)  # 维度1
        return X, y

    def creatmodel(self):
        self.model = Sequential()
        # input_shape 训练长度 每个数据的维度
        self.model.add(SimpleRNN(units=5, input_shape=(self.time_step, 1), activation="relu"))
        # 输出层
        # 输出数值 units =1 1个神经元 "linear"线性模型
        self.model.add(Dense(units=1, activation="linear"))
        # 配置模型 回归模型y
        self.model.compile(optimizer="adam", loss="mean_squared_error")
        self.model.summary()

        self.model.fit(np.array(self.X), np.array(self.y), batch_size=30, epochs=200)

    def predicted_actual_stock_price(self):
        zgpa = ts.get_hist_data('601318', start='2023-01-01', end='2023-07-24')
        # 查看数据前5行
        print(zgpa.head())
        # 输出数据
        # zgpa.to_csv('zgpa_test.csv')
        data_test = zgpa
        price_test = data_test.loc[:, 'close']
        price_test.head()

        price_test_norm = price_test / max(price_test)
        X_test_norm, y_test_norm = self.extract_data(price_test_norm, self.time_step)
        print(X_test_norm.shape, len(y_test_norm))

        y_test_predict = self.model.predict(X_test_norm) * max(price_test)
        y_test = [i * max(price_test) for i in y_test_norm]  #

        fig3 = plt.figure(figsize=(10, 5))
        plt.plot(y_test, label="real price test")
        plt.plot(y_test_predict, label="predict price test")
        plt.title("price")
        plt.xlabel("time")
        plt.ylabel("price")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    mr = MyRnn()
    mr.creatmodel()
    mr.darw()
    mr.predicted_actual_stock_price()
