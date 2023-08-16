import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree

class MyTree:
    '''
    决策树：通过信息熵 排序决策条件，生成决策树。
    '''
    def __init__(self,file):
        # 获取数据文件
        self.data = pd.read_csv(file)
        # 获取训练数据
        self.x = self.data.loc[:, ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
        self.y = self.data.loc[:, 'category']
        #
        self.dc_tree = tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=5)

    def create_model(self):
        self.dc_tree.fit(self.x,self.y)
        tree.plot_tree(self.dc_tree, filled=True, feature_names=['1', '2', '3', '4'], class_names=['0', '1', '2'])

    def draw(self):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定中文字体
        mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
        tree.plot_tree(self.dc_tree,filled=True,feature_names=['1','2','3','4'],class_names=['0','1','2'])

if __name__ == '__main__':
    mt = MyTree('iris.csv')
    mt.create_model()
    mt.draw()