import tushare as ts
# 读取中国平安（601318）数据
zgpa = ts.get_hist_data('601318', start='2020-01-01', end='2020-07-24')
# 查看数据前5行
zgpa.head()
# 输出数据
#zgpa.to_csv('zgpa_test.csv')
data_test = zgpa
price_test = data_test.loc[:,'close']
price_test.head()

price_test_norm = price_test/max(price)
X_test_norm,y_test_norm = extract_data(price_test_norm,time_step)
print(X_test_norm.shape,len(y_test_norm))

y_test_predict = model.predict(X_test_norm) * max(price)
y_test = [i*max(price) for i in y_test_norm]#