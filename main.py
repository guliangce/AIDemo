# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import my_linear_regression
import my_logic_regression
import my_kmeans
import draw_designs

def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。

#线性回归案例实现
def example_linear_regression():
    # 实例1 一个简单的 线性回归 案例。
    lrd = draw_designs.DrawDesigns()
    mlr_ex1 = my_linear_regression.MyLinearRegression('lr_generated_data.csv')
    mlr_ex1.get_x(['x'])
    mlr_ex1.get_y(['y'])
    mlr_ex1.create_model()
    mlr_ex1.check_model()
    lrd.add_sub([mlr_ex1.x,mlr_ex1.y,mlr_ex1.get_predict_y()])

    # 实例2 单因子的房价预测 线性回归 案例。
    mlr_ex2 = my_linear_regression.MyLinearRegression('usa_housing_price.csv')
    mlr_ex2.get_x(['size'])
    mlr_ex2.get_y(['Price'])
    mlr_ex2.create_model()
    mlr_ex2.check_model()
    lrd.add_sub([mlr_ex2.x, mlr_ex2.y, mlr_ex2.get_predict_y()])

    #实例3 多因子的房价预测 线性回归 案例。
    mlr_ex3 = my_linear_regression.MyLinearRegression('usa_housing_price.csv')
    mlr_ex3.get_x(['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Area Population', 'size'])
    mlr_ex3.get_y(['Price'])
    mlr_ex3.create_model()
    lrd.add_sub([mlr_ex3.x, mlr_ex3.y, mlr_ex3.get_predict_y()])
    mlr_ex3.check_model()

    #绘制线性回归的图像。
    lrd.create_sub_designs()
    #绘制一个案例
    lrd.my_scatter(lrd.sub_designs_list[0], mlr_ex1.x, mlr_ex1.y)
    lrd.my_plot(lrd.sub_designs_list[0], mlr_ex1.x, mlr_ex1.get_predict_y())
    #绘制第二个案例
    lrd.my_scatter(lrd.sub_designs_list[1], mlr_ex2.x, mlr_ex2.y)
    lrd.my_plot(lrd.sub_designs_list[1], mlr_ex2.x, mlr_ex2.get_predict_y())
    #绘制第三个案例
    lrd.my_scatter(lrd.sub_designs_list[2], mlr_ex3.y, mlr_ex3.get_predict_y())

    lrd.show()

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':

    #线性回归案例。
    #example_linear_regression()

    #逻辑回归案例。
    '''
    mlor = my_logic_regression.MyLogicRegression('examdata.csv')
    mlor.create_model(mlor.x_second_order)
    print(mlor.get_predicted_data(mlor.x_second_order))
    print(mlor.get_accuracy(mlor.x_second_order))
    print(mlor.get_single_predicted_2(90,90))
    print('你有'+ str(int(mlor.get_accuracy(mlor.x_second_order)*100)) +'%的概率'+mlor.get_single_predicted_2(90,90))
    mlor.draw()
    '''
    #无监督学习
    mul = my_unsupervised_learning.MyUnsupervisedLearning('my_unsupervised_learning.csv')
    mul.draw()