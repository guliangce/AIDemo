import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import  train_test_split
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.metrics import accuracy_score


class MyLSTM:

    def __init__(self):
        # 加载数据
        data = open("LSTM_text.txt").read()
        # 移除换行
        data = data.replace("\n", "").replace("\r", "")
        print(data)
        # 分出字符
        letters = list(set(data))
        print(letters)
        self.num_letters = len(letters)

        # 建立字典
        self.int_to_char = {a: b for a, b in enumerate(letters)}

        char_to_int = {b: a for a, b in enumerate(letters)}
        print(char_to_int)
        # 设置步长
        time_step = 20
        # 提取X y
        X, y = self.data_preprocessing(data, time_step, self.num_letters, char_to_int)
        print(X.shape)
        print(len(y))
        # 训练和测试数据分离
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=10)
        self.y_train_category = to_categorical(self.y_train, self.num_letters)

        #测试数据
        self.new_letters = 'The United States continues to lead the world with more than '
        self.X_new, self.y_new = self.data_preprocessing(self.new_letters, time_step, self.num_letters, char_to_int)



    def createmodel(self):
        self.model = Sequential()
        # input_shape 看样本的
        self.model.add(LSTM(units=20, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), activation="relu"))

        # 输出层 看样本有多少页
        self.model.add(Dense(units=self.num_letters, activation="softmax"))
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model.summary()
        # 训练模型
        self.model.fit(self.X_train, self.y_train_category, batch_size=1000, epochs=50)

    def predict(self):
        # 预测
        y_train_predict = self.model.predict(self.X_train)
        y_train_predict = np.argmax(y_train_predict,axis=1)
        # 转换成文本
        y_train_predict_char = [self.int_to_char[i] for i in y_train_predict]
        print(y_train_predict_char)

        accuracy = accuracy_score(self.y_train, y_train_predict)
        print(accuracy)

        y_test_predict = self.model.predict(self.X_test)
        y_test_predict = np.argmax(y_test_predict, axis=1)
        accuracy_test = accuracy_score(self.y_test, y_test_predict)
        print(accuracy_test)
        y_test_predict_char = [self.int_to_char[i] for i in y_test_predict]

    def test(self):
        y_new_predict = self.model.predict(self.X_new)
        y_new_predict = np.argmax(y_new_predict, axis=1)
        print(y_new_predict)
        y_new_predict_char = [self.int_to_char[i] for i in y_new_predict]
        print(y_new_predict_char)
        for i in range(0, self.X_new.shape[0] - 20):
            print(self.new_letters[i:i + 20], '--predict next letter is --', y_new_predict_char[i])

    def extract_data(self,data,slide):
        x = []
        y = []
        for i in range(len(data) - slide):
            x.append([a for a in data[i:i+slide]])
            y.append(data[i+slide])
        return x,y
    #字符到数字的批量转换
    def char_to_int_Data(self,x,y,char_to_int):
        x_to_int = []
        y_to_int = []
        for i in range(len(x)):
            x_to_int.append([char_to_int[char] for char in x[i]])
            y_to_int.append([char_to_int[char] for char in y[i]])
        return x_to_int,y_to_int

    #实现输入字符文章的批量处理,输入整个字符,滑动窗口大小,转化字典
    def data_preprocessing(self,data,slide,num_letters,char_to_int):
        char_data = self.extract_data(data,slide)
        int_data = self.char_to_int_Data(char_data[0],char_data[1],char_to_int)
        Input = int_data[0]
        Output = list(np.array(int_data[1]).flatten()  )
        Input_RESHAPED = np.array(Input).reshape(len(Input ),slide)
        new = np.random.randint(0,10,size=[Input_RESHAPED.shape[0],Input_RESHAPED.shape[1],num_letters])
        for i in range(Input_RESHAPED.shape[0]):
            for j in range(Input_RESHAPED.shape[1]):
                new[i,j,:] = to_categorical(Input_RESHAPED[i,j],num_classes=num_letters)
        return new,Output


if __name__ == '__main__':
    ml = MyLSTM()
    ml.createmodel()
    ml.predict()
    ml.test()
