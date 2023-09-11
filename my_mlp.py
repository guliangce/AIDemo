import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import accuracy_score

class MyMlp:
    def __init__(self):
        #初始化数据
        path = 'mnist\\mnist.npz'
        (self.x_train,self.y_train),(self.x_test,self.y_test) = mnist.load_data()
        self.img1 = self.x_train[0]

        #将数据转换为 矩阵
        self.size1 = self.img1.shape[0] * self.img1.shape[1]
        x_train_f = self.x_train.reshape(self.x_train.shape[0],self.size1)
        x_test_f = self.x_test.reshape(self.x_test.shape[0],self.size1)

        #归一化
        self.x_train_n = x_train_f / 255
        self.x_test_n = x_test_f / 255

        #输出转换为0 1 的矩阵
        self.y_train_f = to_categorical(self.y_train)
        y_test_f = to_categorical(self.y_test)

        self.mlp = Sequential()
    def create_model_kms(self):

        self.mlp.add(Dense(units=392,activation='sigmoid',input_dim=self.size1))
        self.mlp.add(Dense(units=392,activation='sigmoid'))
        self.mlp.add(Dense(units=10,activation='softmax'))
        self.mlp.summary()
        self.mlp.compile(loss='categorical_crossentropy',optimizer='adam')
        self.mlp.fit(self.x_train_n,self.y_train_f,epochs=10)

    def get_accuracy(self):

        accuracy_tr = accuracy_score(self.y_train,np.argmax(self.mlp.predict(self.x_train_n),axis=1))
        accuracy_te = accuracy_score(self.y_test,np.argmax(self.mlp.predict(self.x_test_n),axis=1))

        print(accuracy_tr,accuracy_te)

    def darw(self):
        img = self.x_test[101]
        fig = plt.figure()
        plt.imshow(img)
        plt.title(np.argmax(self.mlp.predict(self.x_test_n),axis=1)[101])
        plt.show()



if __name__ == '__main__':
    mm = MyMlp()
    mm.create_model_kms()
    mm.get_accuracy()
    mm.darw()