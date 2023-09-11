from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense
from keras.preprocessing.image import load_img,img_to_array
class MyCatDog:
    def __init__(self):
        #加载数据
        train_data = ImageDataGenerator(rescale=1./255)

        self.training_set = train_data.flow_from_directory('./data_set',target_size=(50,50),batch_size=32,class_mode='binary')
        #建立模型
        self.model = Sequential()
    def create_model(self):

        #卷积
        self.model.add(Conv2D(32,(3,3),input_shape=(50,50,3),activation='relu'))
        #池化
        self.model.add(MaxPool2D(pool_size=(2,2)))
        #卷积
        self.model.add(Conv2D(32,(3,3),activation='relu'))
        #池化
        self.model.add(MaxPool2D(pool_size=(2,2)))
        #展开
        self.model.add(Flatten())
        #MLP 多层感知器
        self.model.add(Dense(units=128,activation='relu'))
        self.model.add(Dense(units=1,activation='sigmoid'))
        #配置模型
        self.model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        #展示模型
        self.model.summary()
        #进行训练
        self.model.fit_generator(self.training_set,epochs=25)

    def accuracy_test(self):
        a_t = self.model.evaluate_generator(self.training_set)
        print(a_t)

    def one_img(self,path):
        pic = path
        pic = load_img(pic,target_size=(50,50))
        pic = img_to_array(pic)
        pic = pic/255
        pic = pic.reshape(1,50,50,3)
        res = self.model.predict(pic)

        print(res)


if __name__ == '__main__':
    mcd = MyCatDog()
    mcd.create_model()
    mcd.accuracy_test()
    mcd.one_img('./data_set/cat/cat.0.jpg')
