# Utils for File Operations
import _pickle as pkl
import os

def load_file_with_p(file_path):
    '''载入,p二进制文件'''
    return pkl.load(open(file_path, 'rb'))
    
def save_file_with_p(save_data, file_path):
    '''载入,p二进制文件'''
    pkl.dump(save_data, open(file_path, 'wb'))

def check_file_exits(file_path):
    '''判断文件是否存在'''
    return True if os.path.exists(file_path) else False

def get_dir_info(file_root):
    """获取给定文件夹下的相关信息
    
    Parameters
    ---------
    file_root: (type: str) 文件根目录 \n
    Returns
    ---------
    all_file_paths: 该文件夹下的所有文件绝对路径(type: list) \n
    all_file_names: 该文件夹下的所有文件名 (type: list) \n
    tree_dir: 该文件夹下的文件树(不包括文件名) \n
    """
    # 获取指定目录下的所有文件路径
    all_file_paths = [os.path.join(root, file) for root, dirs, files in os.walk(file_root) for file in files]

    # 获取指定目录下的所有文件名
    all_file_names = [file for root, dirs, files in os.walk(file_root) for file in files]

    # 获取指定目录下的所有文件夹
    all_dir_names = [dir for root, dirs, files in os.walk(file_root) for dir in dirs]

    tree_dir = []
    for file_path in all_file_paths:
        tree_dir.append('/'.join(file_path.replace(file_root,'').split('/')[1:-1]))
    
    return all_file_paths,all_file_names,list(set(tree_dir[1:]))
