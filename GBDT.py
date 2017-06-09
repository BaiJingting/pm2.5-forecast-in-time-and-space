#-*-coding:utf-8 -*-

import pandas as pd

coord_zhandian = pd.read_csv('C:\\Users\\Bai\\Desktop\\data\\监测点坐标.csv')
coord_source = pd.read_csv('C:\\Users\\Bai\\Desktop\\data\\污染源经纬度.csv')


coord_source.iloc[0,:] = coord_source.iloc[0,:] - coord_zhandian.loc[0,'东软']
coord_source.iloc[1,:] = coord_source.iloc[1,:] - coord_zhandian.loc[1,'东软']
coor = list(coord_source.iloc[0])
coor.extend(coord_source.iloc[1])
