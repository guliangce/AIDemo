from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
class MyCatDogVvg:
    def __init__(self):
        self.model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',include_top=False)

        def model_process(img_path):

            img = load_img(img_path,target_size=(224,224))
            img = img_to_array(img)

            x = np.expand_dims(img,axis=0)
            x = preprocess_input(x)
            x_vgg = self.model.predict(x)
            x_vgg = x_vgg.reshape(1,7*7*512)
            return  x_vgg

        import os
        folder = "data_set/cat"
        dirs = os.listdir(folder)
        img_path = []
        for i in dirs:
            if os.path.splitext(i)[1] == ".jpg":
                img_path.append(i)
        img_path = [folder+"//"+i for i in img_path]

        features1 = np.zeros([len(img_path),25088])
        for i in range(len(img_path)):
            feature_i = model_process(img_path[i])
            print('preprocessed:',img_path[i])
            features1[i] = feature_i

        folder = "data_set/dog"
        dirs = os.listdir(folder)
        img_path = []
        for i in dirs:
            if os.path.splitext(i)[1] == ".jpg":
                img_path.append(i)
        img_path = [folder+"//"+i for i in img_path]

        features2 = np.zeros([len(img_path),25088])
        for i in range(len(img_path)):
            feature_i = model_process(img_path[i])
            print('preprocessed:',img_path[i])
            features2[i] = feature_i

        print(features1.shape,features2.shape)
        y1 = np.zeros(12500)
        y2 = np.ones(12500)

        X = np.concatenate((features1,features2),axis=0)
        y = np.concatenate((y1,y2),axis=0)
        y = y.reshape(-1,1)
        print(X.shape,y.shape)

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=50)

        self.model_1 = Sequential()
        self.model_1.add(Dense(units=10,activation='relu',input_dim=25088))
        self.model_1.add(Dense(units=1,activation='sigmoid'))
        self.model_1.summary()

        self.model_1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        self.model_1.fit(X_train,y_train,epochs=50)


if __name__ == '__main__':
    mcdv = MyCatDogVvg()