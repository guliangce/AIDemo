from matplotlib import pyplot as plt

class DrawDesigns:
    def __init__(self):
        self.figure = plt.figure()
        plt.title("AIDemo")
        self.num = 1
        self.designs_list = []
        self.sub_designs_list = []
    def add_sub(self,design):
        '''
        增加子层
        :param design:
        :return:
        '''
        self.designs_list.append(design)

    def my_scatter(self,subplot,x,y):
        subplot.scatter(x,y)

    def my_plot(self,subplot,x,y):
        subplot.plot(x,y,'r')

    def create_sub_designs(self):
        '''
        根据子层添加子画布
        :return:
        '''

        h = len(self.designs_list)
        for i in range(h):
            sp = self.figure.add_subplot(h,1,i+1)
            self.sub_designs_list.append(sp)

    def show(self):
        plt.show()