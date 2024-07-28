# Utils for File Operations

- [source code](../utils/file_op.py)
- Description: This util is used to read / write / process files with python.   
- load_file_with_p : load_file_with_p(file_path) 
- save_file_with_p : save_file_with_p(save_data, file_path)
- check_file_exits : check_file_exits(file_path) : return True or False
- get_dir_info : get_dir_info(file_root) return:
  
    >all_file_paths: Absolute path to all files in this folder(type: list)   
     all_file_names: All file names in the folder (type: list)    
     tree_dir: file tree(不包括文件名) 
     
- check_path_type : check_path_type(path) return : 0:file 1: folder -1:other
