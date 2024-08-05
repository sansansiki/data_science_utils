# utils for data processing

import pandas as pd
from operator import itemgetter
from math import ceil
import numpy as np

def sorted_by_dict_value(dic,desc=False):
    '''对字典的value进行排序\n
    esc:是否升序排序
    '''
    return sorted(dic.items(),key=itemgetter(1),reverse=desc)

def get_df_null_rate(data_new):
    ''' 获取dataframe中每列的缺失值数量和缺失值占比'''
    if type(data_new) == pd.core.series.Series:
        data_new = pd.DataFrame(data_new)
    empty_column = []
    for e, c in enumerate(data_new.columns): 
        empty_column.append(c)
        print("feature_no:%d \t feature_name:%s \t null_num:%d \t null_rate: %.2f%%"% (e, c , sum(pd.isnull(data_new[c])), 
                                                                 100*sum(pd.isnull(data_new[c]))/len(data_new[c])))

def split_dict_into_n_parts(d, n):
    '''将字典按n等份分割'''
    items = list(d.items())
    num_items = len(items)
    items_per_part = ceil(num_items / n)
    
    result = []
    for i in range(0, num_items, items_per_part):
        result.append(dict(items[i:i+items_per_part]))
    
    return result

def remove_highly_correlated_columns(df, threshold):
    '''delete highly correlated(>threshold) columns
    return dataframe and the dropped columns
    '''
    # 计算相关系数矩阵
    corr_matrix = df.corr().abs()

    # 创建一个布尔值的DataFrame，表示相关系数是否大于阈值
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

    # 获取要删除的列名
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    # 删除相关系数较高的列
    df = df.drop(to_drop, axis=1)

    return df,to_drop