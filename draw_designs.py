from matplotlib import pyplot as plt

class DrawDesigns:
    def __init__(self):
        self.figure = plt.figure()
        plt.title("AIDemo")
        self.num = 1
        self.designs_list = []
        self.sub_designs_list = []
    def add_sub(self,design):
        self.designs_list.append(design)

    def my_scatter(self,subplot,x,y):
        subplot.scatter(x,y)

    def my_plot(self,subplot,x,y):
        subplot.plot(x,y)

    def create_sub_designs(self):
        h = len(self.designs_list)
        for i in range(h):
            #print(self.designs_list)
            #print(len(self.designs_list))
            #print(self.designs_list.index(i)+1)
            sp = self.figure.add_subplot(h,1,i+1)
            self.sub_designs_list.append(sp)
            #sp.scatter(self.designs_list[i][0], self.designs_list[i][1])
            #sp.plot(self.designs_list[i][0], self.designs_list[i][2], 'r')
    def show(self):
        plt.show()