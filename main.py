# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import my_linear_regression
import draw_designs
def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
    dd = draw_designs.DrawDesigns()
    # 实例1 一个简单的 线性回归 案例。
    mlr_ex1 = my_linear_regression.MyLinearRegression('lr_generated_data.csv')
    mlr_ex1.get_x(['x'])
    mlr_ex1.get_y(['y'])
    mlr_ex1.create_model()
    mlr_ex1.check_model()
    dd.add_sub([mlr_ex1.x,mlr_ex1.y,mlr_ex1.get_predict_y()])

    #mlr_ex1.draw()

    # 实例2 单因子的房价预测 线性回归 案例。
    mlr_ex2 = my_linear_regression.MyLinearRegression('usa_housing_price.csv')
    mlr_ex2.get_x(['size'])
    mlr_ex2.get_y(['Price'])
    mlr_ex2.create_model()
    mlr_ex2.check_model()
    dd.add_sub([mlr_ex2.x, mlr_ex2.y, mlr_ex2.get_predict_y()])
    #mlr_ex2.draw()

    #实例3 多因子的房价预测 线性回归 案例。
    mlr_ex3 = my_linear_regression.MyLinearRegression('usa_housing_price.csv')
    mlr_ex3.get_x(['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Area Population','size'])
    mlr_ex3.get_y(['Price'])
    mlr_ex3.create_model()
    dd.add_sub([mlr_ex3.x, mlr_ex3.y, mlr_ex3.get_predict_y()])
    mlr_ex3.check_model()

    dd.create_sub_designs()

    dd.my_scatter(dd.sub_designs_list[0],mlr_ex1.x,mlr_ex1.y)
    #dd.my_plot(dd.sub_designs_list[0], mlr_ex1.x,mlr_ex1.get_predict_y)

    dd.my_scatter(dd.sub_designs_list[1], mlr_ex2.x, mlr_ex2.y)
    #dd.my_plot(dd.sub_designs_list[1], mlr_ex2.x, mlr_ex2.get_predict_y)

    dd.my_scatter(dd.sub_designs_list[2], mlr_ex3.y, mlr_ex3.get_predict_y())

    dd.show()