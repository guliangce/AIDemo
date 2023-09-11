import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import joblib

class MySecTrain:
    def __init__(self):
        def fix_seed(seed=1):
            # reproducible
            np.random.seed(seed)

        # make up data建立数据
        fix_seed(1)
        x_data = np.linspace(-5, 5, 100)[:, np.newaxis]  # 水平轴-7~10
        print(x_data)
        noise = np.random.normal(0, 0.1, x_data.shape)
        self.y_data = np.square(x_data) - 5 + noise

        # 维度转换
        self.X = x_data.reshape(-1, 1)

    def create_model(self):

        self.model1 = Sequential()
        self.model1.add(Dense(units=50, input_dim=1, activation="relu"))
        self.model1.add(Dense(units=50, activation="relu"))
        self.model1.add(Dense(units=1, activation="linear"))
        self.model1.compile(optimizer="adam", loss="mean_squared_error")
        self.model1.summary()
        # 训练模型
        self.model1.fit(self.X, self.y_data, epochs=400)

        # 预测
        y_predict = self.model1.predict(self.X)
        print(y_predict)

        #可视化
        fig2 = plt.figure(figsize=(7, 5))
        plt.scatter(self.X, self.y_data)
        plt.plot(self.X, y_predict, 'g')
        plt.title('y-x 100 predict')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    def save_model(self):

        joblib.dump(self.model1, 'model.pkl')

    def load_model(self):
        model2 = joblib.load('model.pkl')
        return model2

    def test_new_data(self):
        def fix_seed(seed=1):
            # reproducible
            np.random.seed(seed)

        fix_seed(1)
        X2 = np.linspace(-6, 6, 120)[:, np.newaxis] + [2]  # 水平轴 + 偏移
        print(X2)
        # np.random.shuffle(x_data)
        noise = np.random.normal(1, 0.5, X2.shape)
        y_data2 = np.square(X2) - 10 + noise
        print(y_data2)

        model = self.load_model()
        y2_predict = model.predict(X2)

        fig3 = plt.figure(figsize=(7, 5))
        plt.plot(X2, y2_predict, 'r')
        plt.scatter(self.X, self.y_data, label="data1")
        plt.scatter(X2, y_data2, label="data2")
        plt.title('model predict')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

        model.fit(X2, y_data2, epochs=10)
        y2_predict2 = model.predict(X2)

        fig4 = plt.figure(figsize=(7, 5))
        plt.plot(X2, y2_predict2, 'r')
        plt.scatter(self.X, self.y_data, label="data1")
        plt.scatter(X2, y_data2, label="data2")
        plt.title('model predict epochs = 10')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    mst = MySecTrain()
    mst.create_model()
    mst.save_model()
    mst.test_new_data()
