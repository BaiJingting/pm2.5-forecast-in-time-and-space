#-*-coding:utf-8-*-

import load_data as ld, numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation

data_X, data_Y = ld.LoadData()

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


def scorefun(y_pred, y_true):
    output_errors = np.average(abs(y_true - y_pred) , axis=0)  ##*1.0/y_true
    return output_errors

score = scorefun(Y_pred,Y_test)
print(score)