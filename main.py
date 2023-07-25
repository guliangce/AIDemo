# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。

def my_print(*args):
    # 用于调试，不使用直接pass
    #pass
    print(*args)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')

    #打开文件 lr_generated_data.csv
    data = pd.read_csv('lr_generated_data.csv')

    #获取文件的X 和 Y 列
    x = data.loc[:,['x']]
    y = data.loc[:,['y']]
    my_print(x.shape)

    #创建线性模型
    lr_model = LinearRegression()
    my_print(x,y)
    lr_model.fit(x,y)
    my_print(lr_model)

    #预测
    y_p = lr_model.predict(x)
    my_print(y_p)

    #模型参数
    my_print("模型参数")
    my_print(lr_model.coef_)
    my_print(lr_model.intercept_)

    #测量模型精度
    MSE = mean_squared_error(y,y_p)
    R2 = r2_score(y,y_p)
    my_print(MSE,R2)

    #创建散点图
    plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x,y)
    plt.plot(x,y_p)
    plt.show()



# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
