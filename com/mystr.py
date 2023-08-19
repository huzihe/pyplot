'''
Author: hzh huzihe@whu.edu.cn
Date: 2023-08-13 21:05:21
LastEditTime: 2023-08-13 21:36:30
FilePath: /pyplot/com/mystr.py
Descripttion: 
'''
def replace_char(old_string, char, index):
    """
    字符串按索引位置替换字符
    """
    old_string = str(old_string)
    # 新的字符串 = 老字符串[:要替换的索引位置] + 替换成的目标字符 + 老字符串[要替换的索引位置+1:]
    new_string = old_string[:index] + char + old_string[index + 1 :]
    return new_string

def add_char(old_string, char, index):
    '''
    将字符串按索引位置添加字符
    '''
    old_string = str(old_string)
    # 新的字符串 = 老字符串[:要替换的索引位置] + 替换成的目标字符 + 老字符串[要替换的索引位置+1:]
    new_string = old_string[:index] + char + old_string[index:]
    return new_string