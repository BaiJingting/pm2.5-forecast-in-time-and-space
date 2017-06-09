#-*-coding:utf-8-*-

import pandas as pd, numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error

climate = pd.read_csv('C:\\Users\\Bai\\Desktop\\data\\天气.csv')
climate.drop(['日期','湿度分类','风速（m/s）','风向'],axis=1,inplace=True)
pm2_5 = pd.read_csv('C:\\Users\\Bai\\Desktop\\data\\pm25.csv')
pollute = pd.read_csv('C:\\Users\\Bai\\Desktop\\data\\污染源颗粒物数据.csv')
pollute.drop(['时间'],axis=1,inplace=True)
coord_zhandian = pd.read_csv('C:\\Users\\Bai\\Desktop\\data\\监测点坐标.csv')
coord_source = pd.read_csv('C:\\Users\\Bai\\Desktop\\data\\污染源经纬度.csv')

k=1
for zhandian in (list(pm2_5.columns)[1:12]):
    coor = []
    coor.extend(list(coord_source.iloc[0, :] - coord_zhandian.loc[0, zhandian]))
    coor.extend(list(coord_source.iloc[1, :] - coord_zhandian.loc[1, zhandian]))
    ## 经纬度差值使用平方和指数形式放进模型，在测试集上的损失无明显变化，也印证了GBDT在特征工程方面的优势。但gbdt在模型特征的自定义上没有lr好
    # coor.extend(list((coord_source.iloc[0, :] - coord_zhandian.loc[0, zhandian])**2))
    # coor.extend(list((coord_source.iloc[1, :] - coord_zhandian.loc[1, zhandian])**2))
    print(coor)
    Coor = [];
    for i in range(climate.shape[0]):
        Coor.append(coor)
    data = pd.concat([climate, pollute.shift(1), pollute.shift(2), pollute.shift(3), pd.DataFrame(Coor), pm2_5[[zhandian]]], axis=1)
    data = data.dropna()
    data.columns = [j for j in range(data.columns.shape[0])]
    if(k==1): Data = data;
    else: Data = pd.concat([Data, data],axis=0)
    k = k+1
print(Data.shape)

data_X = Data.iloc[:,:168]
data_Y = Data.iloc[:,168]

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(data_X, data_Y, test_size=0.2, random_state=0)

## 迭代次数在300之后损失无明显下降
# for m in range(5):
#     iter = (m+1)*100
#     GBR = GradientBoostingRegressor(n_estimators=iter, learning_rate=0.02, max_depth=10, subsample=0.8, loss='ls').fit(X_train, Y_train)
#     Y_pred = GBR.predict(X_test)
#     score = mean_squared_error(Y_pred,Y_test)
#     print(iter, score)

## 树最大深度为9效果最好
# for m in range(10):
#     GBR = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=m+1, subsample=0.8, loss='ls').fit(X_train, Y_train)
#     Y_pred = GBR.predict(X_test)
#     score = mean_squared_error(Y_pred,Y_test)
#     print(m+1, score)

## 学习率0.1最好
# for m in range(10):
#     rate = 0.1-m*0.01
#     GBR = GradientBoostingRegressor(n_estimators=300, learning_rate=rate, max_depth=7, subsample=0.8, loss='ls').fit(X_train, Y_train)
#     Y_pred = GBR.predict(X_test)
#     score = mean_squared_error(Y_pred,Y_test)
#     print(rate, score)

GBR = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=9, subsample=0.8, loss='ls').fit(X_train, Y_train)
Y_pred = GBR.predict(X_test)
score = mean_squared_error(Y_pred,Y_test)
print(score)