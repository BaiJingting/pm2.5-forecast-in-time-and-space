#-*-coding:utf-8-*-

import load_data as ld, numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV

data_X, data_Y = ld.LoadData()

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(data_X, data_Y, test_size=0.2, random_state=0)

def scorefun(y_pred, y_true):
    output_errors = np.average(abs(y_true - y_pred) , axis=0)  ##*1.0/y_true
    return output_errors


GBR = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=9, subsample=0.8, max_features=0.5, loss='ls')

## 网格搜索找到最优参数
tuned_parameter = [{'n_estimators':[100,200,300], 'max_depth':[6,7,8,9], 'subsample':[0.7,0.8,0.9,1], 'max_features':[0.6,0.7,0.8,0.9,1]}]
model = GridSearchCV(GBR, tuned_parameter, cv=5)
model.fit(X_train, Y_train)
print("Best parameters set found: ")
print(model.best_params_)

# Y_pred = GBR.predict(X_test)
# score = scorefun(Y_pred,Y_test)
# print(score)