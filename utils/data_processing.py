# utils for data processing

import pandas as pd
from operator import itemgetter


def sorted_by_dict_value(dic,desc=False):
    '''对字典的value进行排序\n
    esc:是否升序排序
    '''
    return sorted(dic.items(),key=itemgetter(1),reverse=desc)

