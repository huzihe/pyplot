'''
Author: hzh huzihe@whu.edu.cn
Date: 2025-01-25 23:34:34
LastEditTime: 2025-01-27 09:47:07
FilePath: /pyplot/com/panda2csv.py
Descripttion: 
'''
import csv

def text_to_csv(text_file, csv_file):
  """
  将文本文件转换为 CSV 文件

  Args:
    text_file: 文本文件名
    csv_file: CSV文件名
  """
  data = []
  with open(text_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    
    # 找到 "END OF HEADER" 行的索引
    end_header_index = next((i for i, line in enumerate(lines) if line.startswith("##-> END OF HEADER")), None)
    if end_header_index is None:
        raise ValueError("文件头部分未找到 '##-> END OF HEADER' 标记")
    
    data_lines = lines[end_header_index + 1:]

    for line in data_lines:
      # 根据文本文件的分隔符分割行数据
      # 这里假设分隔符为逗号，可根据实际情况修改
      row = line.strip().split()
      data.append(row)
  
  with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(data)

# 示例用法
# text_file = './data/202401/PPP/pos_2024020_ally_3d_if'
# csv_file = './data/202401/PPP/pos_2024020_ally_3d_if.csv'
# text_to_csv(text_file, csv_file)

# text_file = './data/202401/PPP/pos_2024020_ally_3d_if_dw'
# csv_file = './data/202401/PPP/pos_2024020_ally_3d_if_dw.csv'
# text_to_csv(text_file, csv_file)

# text_file = './data/202401/PPP/pos_2024020_ally_rt_if_del'
# csv_file = './data/202401/PPP/pos_2024020_ally_rt_if_del.csv'
# text_to_csv(text_file, csv_file)

# text_file = './data/202401/PPP/pos_2024020_ally_rt_if'
# csv_file = './data/202401/PPP/pos_2024020_ally_rt_if.csv'
# text_to_csv(text_file, csv_file)

# text_file = './data/202401/PPP/pos_2024020_ally_rt_if_dw'
# csv_file = './data/202401/PPP/pos_2024020_ally_rt_if_dw.csv'
# text_to_csv(text_file, csv_file)

# text_file = './data/202401/PPP/pos_2024020_k_ally_3d_if'
# csv_file = './data/202401/PPP/pos_2024020_k_ally_3d_if.csv'
# text_to_csv(text_file, csv_file)

# text_file = './data/202401/PPP/pos_2024020_k_ally_3d_if_dw'
# csv_file = './data/202401/PPP/pos_2024020_k_ally_3d_if_dw.csv'
# text_to_csv(text_file, csv_file)

# text_file = './data/202401/PPP/pos_2024020_k_ally_3d_if_del'
# csv_file = './data/202401/PPP/pos_2024020_k_ally_3d_if_del.csv'
# text_to_csv(text_file, csv_file)

# text_file = './data/202401/PPP/pos_2024020_k_ally_rt_if_del'
# csv_file = './data/202401/PPP/pos_2024020_k_ally_rt_if_del.csv'
# text_to_csv(text_file, csv_file)

# text_file = './data/202401/PPP/pos_2024020_k_ally_rt_if'
# csv_file = './data/202401/PPP/pos_2024020_k_ally_rt_if.csv'
# text_to_csv(text_file, csv_file)

# text_file = './data/202401/PPP/pos_2024020_k_ally_rt_if_dw'
# csv_file = './data/202401/PPP/pos_2024020_k_ally_rt_if_dw.csv'
# text_to_csv(text_file, csv_file)

text_file = './data/202401/PPP/pos_2024020_k_ublx_3d_if'
csv_file = './data/202401/PPP/pos_2024020_k_ublx_3d_if.csv'
text_to_csv(text_file, csv_file)

text_file = './data/202401/PPP/pos_2024020_k_ublx_3d_if_dw'
csv_file = './data/202401/PPP/pos_2024020_k_ublx_3d_if_dw.csv'
text_to_csv(text_file, csv_file)

text_file = './data/202401/PPP/pos_2024020_k_ublx_3d_if_del'
csv_file = './data/202401/PPP/pos_2024020_k_ublx_3d_if_del.csv'
text_to_csv(text_file, csv_file)

text_file = './data/202401/PPP/pos_2024020_k_ublx_rt_if_del'
csv_file = './data/202401/PPP/pos_2024020_k_ublx_rt_if_del.csv'
text_to_csv(text_file, csv_file)

text_file = './data/202401/PPP/pos_2024020_k_ublx_rt_if'
csv_file = './data/202401/PPP/pos_2024020_k_ublx_rt_if.csv'
text_to_csv(text_file, csv_file)

text_file = './data/202401/PPP/pos_2024020_k_ublx_rt_if_dw'
csv_file = './data/202401/PPP/pos_2024020_k_ublx_rt_if_dw.csv'
text_to_csv(text_file, csv_file)


