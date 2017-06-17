#-*-coding:utf-8-*-
from sklearn.linear_model import Ridge, Lasso, LinearRegression

import load_data as ld, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation
import pickle
from sklearn.preprocessing import OneHotEncoder

def scorefun(y_pred, y_true):
    output_errors = np.average(abs(y_true - y_pred) , axis=0)  ##*1.0/y_true
    return output_errors

def train(X_train, Y_train, path):
    with open(path, 'wb') as f:
        ## 最优参数由GBR.py里网格搜索得到
        GBR = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=9, subsample=0.8, max_features=0.8, loss='ls').fit(X_train, Y_train)
        pickle.dump(GBR, f)

def loadModel(path):
    with open(path, 'rb') as f:
        GBR = pickle.load(f)
    return GBR

if(__name__ == "__main__"):
    path = "model.pkl"
    data_X, data_Y = ld.LoadData()
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(data_X, data_Y, test_size=0.2, random_state=0)

    train(X_train, Y_train, path)
    GBR = loadModel(path)

    data1 = GBR.apply(X_train)
    data2 = GBR.apply(X_test)
    rows = data1.shape[0]

    OneHot = OneHotEncoder()
    X_trans = OneHot.fit_transform(np.concatenate([data1,data2], axis=0))

    data_train = X_trans[:rows,:]
    data_test = X_trans[rows:,:]

    model = Ridge(alpha=0.1, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None).fit(data_train, Y_train)    ##误差 9.5
    # model = Lasso(alpha=0.1, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic').fit(data_train, Y_train)   ## 误差 15
    # model = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=1).fit(data_train, Y_train)  ## 效果更差，误差 28
    Y_pred = model.predict(data_test)

    score = scorefun(Y_pred, Y_test)
    print(score)
