from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift,estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift,estimate_bandwidth


class MyApple:
    def __init__(self):
        path = """C:\\Users\Administrator\PycharmProjects\pythonProject\AIDemo\\data_set\\apple\\original_data"""
        dst_path = """C:\\Users\Administrator\PycharmProjects\pythonProject\AIDemo\\data_set\\apple\\gen_data"""

        datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                                     height_shift_range=0.02, horizontal_flip=True,
                                     vertical_flip=True)
        gen = datagen.flow_from_directory(path, target_size=(224, 224),
                                          batch_size=2, save_to_dir=dst_path,
                                          save_prefix="gen", save_format="jpg")
        # 224,224 VGG 输入的大小
        for i in range(100):
            gen.next()

    def feature_extraction(self):

        print("1.通过VGG16提出图片特征向量！.........................")
        self.model_vgg = VGG16(weights="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top=False)

        folder = """C:\\Users\Administrator\PycharmProjects\pythonProject\AIDemo\\data_set\\apple\\train_data"""
        dirs = os.listdir(folder)
        print(dirs)
        img_path = []
        for i in dirs:
            # if os.path.splitext(i)[1]==".jpg":
            img_path.append(i)
        img_path = [folder + "\\" + i for i in img_path]
        print(img_path)
        self.ip = img_path
        def modelProcess(img_path, model):
            img = load_img(img_path, target_size=(224, 224))
            img = img_to_array(img)
            X = np.expand_dims(img, axis=0)  # 增加一个维度
            X = preprocess_input(X)
            X_VGG = model.predict(X)
            X_VGG = X_VGG.reshape(1, 7 * 7 * 512)
            return X_VGG

        features_train = np.zeros([len(img_path), 7 * 7 * 512])
        for i in range(len(img_path)):
            features_i = modelProcess(img_path[i], self.model_vgg)
            print("preprocessed:", img_path[i])
            features_train[i] = features_i
        print("Done")
        print(features_train.shape)

        self.X = features_train

    def kmaens(self):

        print("2.尝试使用kmean进行预测.......................")
        cnn_kmeans = KMeans(n_clusters=2, max_iter=2000)
        cnn_kmeans.fit(self.X)

        y_predict_kmeans = cnn_kmeans.predict(self.X)
        print(y_predict_kmeans)
        print(Counter(y_predict_kmeans))

        normal_apple_id = 1
        fig2 = plt.figure(figsize=(10, 40))
        for i in range(40):
            for j in range(5):
                img = load_img(self.ip[i * 5 + j])
                plt.subplot(45, 5, i * 5 + j + 1)
                plt.title("apple" if y_predict_kmeans[i * 5 + j] == normal_apple_id else "other")
                plt.imshow(img)
                plt.axis("off")

        import os
        folder_test = """C:\\Users\Administrator\PycharmProjects\pythonProject\AIDemo\\data_set\\apple\\test_data"""
        dirs_test = os.listdir(folder_test)
        img_path_test = []
        for i in dirs_test:
            # if os.path.splitext(i)[1]==".jpg":
            img_path_test.append(i)
        img_path_test = [folder_test + "\\" + i for i in img_path_test]
        print(img_path_test)


        def modelProcess(img_path, model):
            img = load_img(img_path, target_size=(224, 224))
            img = img_to_array(img)
            X = np.expand_dims(img, axis=0)  # 增加一个维度
            X = preprocess_input(X)
            X_VGG = model.predict(X)
            X_VGG = X_VGG.reshape(1, 7 * 7 * 512)
            return X_VGG

        # 数据处理
        features_test = np.zeros([len(img_path_test), 7 * 7 * 512])
        for i in range(len(img_path_test)):
            features_i = modelProcess(img_path_test[i], self.model_vgg)
            print("preprocessed:", img_path_test[i])
            features_test[i] = features_i
        print("Done")

        self.X_test = features_test
        print(self.X_test.shape)

        y_predict_kmeans_test = cnn_kmeans.predict(self.X_test)
        print(y_predict_kmeans_test)

        fig3 = plt.figure(figsize=(10, 10))
        for i in range(1):
            for j in range(3):
                img = load_img(img_path_test[i * 4 + j])
                plt.subplot(3, 4, i * 4 + j + 1)
                plt.title("apple" if y_predict_kmeans[i * 4 + j] == normal_apple_id else "other")
                plt.imshow(img)
                plt.axis("off")

    def meanshift(self):

        print("3.尝试meanshift算法...........................")
        bw = estimate_bandwidth(self.X, n_samples=140)
        print(bw)

        cnn_ms = MeanShift(bandwidth=bw)
        cnn_ms.fit(self.X)

        y_predict_ms = cnn_ms.predict(self.X)
        print(y_predict_ms)


        print(Counter(y_predict_ms))

        normal_apple_id = 0
        fig4 = plt.figure(figsize=(10, 40))
        for i in range(40):
            for j in range(5):
                img = load_img(self.ip[i * 5 + j])
                plt.subplot(45, 5, i * 5 + j + 1)
                plt.title("apple" if y_predict_ms[i * 5 + j] == normal_apple_id else "other")
                plt.imshow(img)
                plt.axis("off")

        y_predict_ms_test = cnn_ms.predict(self.X_test)
        print(y_predict_ms_test)

    def pca(self):
        print("4.降维，降低噪音后再使用meanshift！.......................")
        stds = StandardScaler()
        X_norm = stds.fit_transform(self.X)
        # PCA analusis
        from sklearn.decomposition import PCA
        pca = PCA(n_components=200)
        X_pca = pca.fit_transform(X_norm)
        var_ratio = pca.explained_variance_ratio_
        print(np.sum(var_ratio))
        print(X_pca.shape, self.X.shape)

        bw = estimate_bandwidth(X_pca, n_samples=140)  # 用处理后的数据
        print(bw)

        cnn_pca_ms = MeanShift(bandwidth=bw)
        cnn_pca_ms.fit(X_pca)
        MeanShift(bandwidth=143.03822268262002)
        y_predict_pca_ms = cnn_pca_ms.predict(X_pca)
        print(y_predict_pca_ms)
        print(Counter(y_predict_pca_ms))
        #普通苹果的标识
        normal_apple_id = 0

        # 数据转换
        X_norm_test = stds.transform(self.X_test)
        X_pca_test = pca.transform(X_norm_test)

        # 预测 测试集
        y_predict_pca_ms_test = cnn_pca_ms.predict(X_pca_test)
        print(y_predict_pca_ms_test)



if __name__ == "__main__":
    ma = MyApple()
    ma.feature_extraction()
    ma.kmaens()
    ma.meanshift()
    ma.pca()