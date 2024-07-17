# 文件操作的工具类
import _pickle as pkl

def load_file_with_p(file_path):
    '''载入,p二进制文件'''
    return pkl.load(open(file_path, 'rb'))
    
def save_file_with_p(save_data, file_path):
    '''载入,p二进制文件'''
    pkl.dump(save_data, open(file_path, 'wb'))