#-*-coding:utf-8 -*-

import numpy as np, load_data as ld

from sklearn import cross_validation
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

data_X, data_Y = ld.LoadData_Regression()

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(data_X, data_Y, test_size=0.2, random_state=0)

## 数据存在严重的多重共线性，直接用线性回归效果不好
# model = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=1).fit(X_train,Y_train)

## alpha<0.1都还行，但跟普通线性回归区别不大
## 再试fun为一次或二次函数
# for i in range(100):
#     alpha = 0.01+0.01*i
#     model = Ridge(alpha=alpha, fit_intercept=True, normalize=True, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None).fit(X_train,Y_train)
#     Y_pred = model.predict(X_test)
#     score = mean_squared_error(Y_pred,Y_test)
#     print(alpha, score)

model = Ridge(alpha=0.1, fit_intercept=True, normalize=True, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None).fit(X_train,Y_train)

# model = Lasso(alpha=0.1, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic').fit(X_train,Y_train)
Y_pred = model.predict(X_test)

def scorefun(y_pred, y_true):
    output_errors = np.average(abs(y_true - y_pred) , axis=0)  ##*1.0/y_true
    return output_errors

score = scorefun(Y_pred,Y_test)
print(score)
