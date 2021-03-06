# -*-coding:utf-8 -*-

import pandas as pd
from math import sqrt, exp

def LoadData():
    climate = pd.read_csv('C:\\Users\\Bai\\Desktop\\data\\天气.csv')

    # climate['周一'] = climate.apply(lambda x: 1 if pd.to_datetime(x['日期']).weekday()==0 else 0, axis=1)
    # climate['周二'] = climate.apply(lambda x: 1 if pd.to_datetime(x['日期']).weekday()==1 else 0, axis=1)
    # climate['周三'] = climate.apply(lambda x: 1 if pd.to_datetime(x['日期']).weekday()==2 else 0, axis=1)
    # climate['周四'] = climate.apply(lambda x: 1 if pd.to_datetime(x['日期']).weekday()==3 else 0, axis=1)
    # climate['周五'] = climate.apply(lambda x: 1 if pd.to_datetime(x['日期']).weekday()==4 else 0, axis=1)
    # climate['周六'] = climate.apply(lambda x: 1 if pd.to_datetime(x['日期']).weekday()==5 else 0, axis=1)
    # climate['周日'] = climate.apply(lambda x: 1 if pd.to_datetime(x['日期']).weekday()==6 else 0, axis=1)

    # climate['工作日'] = climate.apply(lambda x: 1 if pd.to_datetime(x['日期']).weekday()<5 else 0, axis=1)

    climate.drop(['日期', '湿度分类', '风速（m/s）', '风向'], axis=1, inplace=True)
    pm2_5 = pd.read_csv('C:\\Users\\Bai\\Desktop\\data\\pm25.csv')
    pollute = pd.read_csv('C:\\Users\\Bai\\Desktop\\data\\污染源颗粒物数据.csv')
    pollute.drop(['时间'], axis=1, inplace=True)
    coord_zhandian = pd.read_csv('C:\\Users\\Bai\\Desktop\\data\\监测点坐标.csv')
    coord_source = pd.read_csv('C:\\Users\\Bai\\Desktop\\data\\污染源经纬度.csv')

    k = 1
    for zhandian in (list(pm2_5.columns)[1:12]):
        coor = []
        coor.extend(list(coord_source.iloc[0, :] - coord_zhandian.loc[0, zhandian]))
        coor.extend(list(coord_source.iloc[1, :] - coord_zhandian.loc[1, zhandian]))
        ## 经纬度差值使用平方和指数形式放进模型，在测试集上的损失无明显变化，也印证了GBDT在特征工程方面的优势。但gbdt在模型特征的自定义上没有lr好
        # coor.extend(list((coord_source.iloc[0, :] - coord_zhandian.loc[0, zhandian])**2))
        # coor.extend(list((coord_source.iloc[1, :] - coord_zhandian.loc[1, zhandian])**2))
        Coor = [];
        for i in range(climate.shape[0]):
            Coor.append(coor)
        data = pd.concat(
            [climate, pollute.shift(1), pollute.shift(2), pollute.shift(3), pd.DataFrame(Coor), pm2_5[[zhandian]]],
            axis=1)
        data = data.dropna()
        data.columns = [j for j in range(data.columns.shape[0])]
        if (k == 1):
            Data = data;
        else:
            Data = pd.concat([Data, data], axis=0)
        k = k + 1

    data_X = Data.iloc[:, :168]
    data_Y = Data.iloc[:, 168]

    return data_X, data_Y

def fun(x):
    return exp(x) #sqrt(x), exp(x), x, x**2

def LoadData_Regression():
    climate = pd.read_csv('C:\\Users\\Bai\\Desktop\\data\\天气.csv')
    climate.drop(['日期', '湿度分类', '风速（m/s）', '风向'], axis=1, inplace=True)
    pollute = pd.read_csv('C:\\Users\\Bai\\Desktop\\data\\污染源颗粒物数据.csv')
    pollute.drop(['时间'], axis=1, inplace=True)
    coord_zhandian = pd.read_csv('C:\\Users\\Bai\\Desktop\\data\\监测点坐标.csv')
    coord_source = pd.read_csv('C:\\Users\\Bai\\Desktop\\data\\污染源经纬度.csv')

    pm2_5 = pd.read_csv('C:\\Users\\Bai\\Desktop\\data\\pm25.csv')

    k = 1
    for zhandian in list(pm2_5.columns)[1:12]:
        lat = list(coord_source.iloc[0, :] - coord_zhandian.loc[0, zhandian])
        lon = list(coord_source.iloc[1, :] - coord_zhandian.loc[1, zhandian])
        data = pd.concat([climate, pollute.shift(1), pollute.shift(2), pollute.shift(3), pm2_5[[zhandian]]], axis=1)  ##
        data.columns = [j for j in range(data.columns.shape[0])]

        for i in range(3):
            data.iloc[:, (6 + i * 44):(7 + i * 44)] = data.iloc[:, (6 + i * 44):(7 + i * 44)] / fun(lat[0] ** 2 + lon[0] ** 2)
            data.iloc[:, (8 + i * 44):(9 + i * 44)] = data.iloc[:, (8 + i * 44):(9 + i * 44)] / fun(lat[1] ** 2 + lon[1] ** 2)
            data.iloc[:, (10 + i * 44):(12 + i * 44)] = data.iloc[:, (10 + i * 44):(12 + i * 44)] / fun(lat[2] ** 2 + lon[2] ** 2)
            data.iloc[:, (13 + i * 44):(18 + i * 44)] = data.iloc[:, (13 + i * 44):(18 + i * 44)] / fun(lat[3] ** 2 + lon[3] ** 2)
            data.iloc[:, (19 + i * 44)] = data.iloc[:, (19 + i * 44)] / fun(lat[4] ** 2 + lon[4] ** 2)
            data.iloc[:, (20 + i * 44):(22 + i * 44)] = data.iloc[:, (20 + i * 44):(22 + i * 44)] / fun(lat[5] ** 2 + lon[5] ** 2)
            data.iloc[:, (23 + i * 44):(24 + i * 44)] = data.iloc[:, (23 + i * 44):(24 + i * 44)] / fun(lat[6] ** 2 + lon[6] ** 2)
            data.iloc[:, (25 + i * 44):(26 + i * 44)] = data.iloc[:, (25 + i * 44):(26 + i * 44)] / fun(lat[7] ** 2 + lon[7] ** 2)
            data.iloc[:, (27 + i * 44):(28 + i * 44)] = data.iloc[:, (27 + i * 44):(28 + i * 44)] / fun(lat[8] ** 2 + lon[8] ** 2)
            data.iloc[:, (29 + i * 44):(33 + i * 44)] = data.iloc[:, (29 + i * 44):(33 + i * 44)] / fun(lat[9] ** 2 + lon[9] ** 2)
            data.iloc[:, (34 + i * 44):(35 + i * 44)] = data.iloc[:, (34 + i * 44):(35 + i * 44)] / fun(lat[10] ** 2 + lon[10] ** 2)
            data.iloc[:, (36 + i * 44)] = data.iloc[:, (36 + i * 44)] / fun(lat[11] ** 2 + lon[11] ** 2)
            data.iloc[:, (37 + i * 44):(44 + i * 44)] = data.iloc[:, (37 + i * 44):(44 + i * 44)] / fun(lat[12] ** 2 + lon[12] ** 2)
            data.iloc[:, (45 + i * 44):(47 + i * 44)] = data.iloc[:, (45 + i * 44):(47 + i * 44)] / fun(lat[13] ** 2 + lon[13] ** 2)
            data.iloc[:, (48 + i * 44):(49 + i * 44)] = data.iloc[:, (48 + i * 44):(49 + i * 44)] / fun(lat[14] ** 2 + lon[14] ** 2)
        data = data.dropna()
        data.columns = [j for j in range(data.columns.shape[0])]
        if (k == 1):
            Data = data;
        else:
            Data = pd.concat([Data, data], axis=0)
        k = k + 1

    print(Data.shape)

    data_X = Data.iloc[:, :138]
    data_Y = Data.iloc[:, 138]

    return data_X, data_Y