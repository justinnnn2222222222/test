import os
import shutil
import re
from datetime import datetime
from matplotlib.font_manager import FontProperties
import time

import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.font_manager import FontProperties
import json
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import csv
import pandas as pd




def extract_datetime_from_path(path):
    # 正则表达式匹配日期和时间部分
    match = re.search(r'(\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})', path)
    if match:
        date_str = match.group(1)
        # 将日期和时间部分转换为 datetime 对象
        return datetime.strptime(date_str, '%Y_%m_%d-%H_%M_%S')
    return None

def find_latest_path(file_paths):
    latest_path = None
    latest_datetime = None
    
    for path in file_paths:
        path_datetime = extract_datetime_from_path(path)
        if path_datetime:
            if not latest_datetime or path_datetime > latest_datetime:
                latest_datetime = path_datetime
                latest_path = path
    
    return latest_path


def find_files(directory, extension):
    try:
        files = [os.path.join(root, filename) for root, dirs, filenames in os.walk(directory)
                 for filename in filenames if filename.endswith(extension)]
        return files
    except Exception as e:
        print(f"查找文件时出错: {e}")
        return []

def filter_files(file_paths, include_keyword=None, exclude_keyword=None):
    try:
        filtered_paths = file_paths
        if include_keyword:
            filtered_paths = [path for path in filtered_paths if all(keyword in path for keyword in include_keyword)]
            # filtered_paths = [path for path in filtered_paths if include_keyword in path]
        if exclude_keyword:
            filtered_paths = [path for path in filtered_paths if all(keyword not in path for keyword in exclude_keyword)]
            # filtered_paths = [path for path in filtered_paths if exclude_keyword not in path]
        return filtered_paths
    except Exception as e:
        print(f"过滤文件时出错: {e}")
        return []

def sort_files(file_paths, keywords):
    sorted_files = []
    for keyword in keywords:
        filtered_files = [path for path in file_paths if keyword in path]
        sorted_files.extend(filtered_files)
    return sorted_files


def get_latest_paths(file_paths, includes):
    latest_paths = []
    for include_config in includes:
        filtered_files = filter_files(file_paths, include_config, None)
        print(f"筛选出的路径列表为: {filtered_files}")
        # 找出最新的路径
        latest_path = find_latest_path(filtered_files)
        if latest_path:
            latest_paths.append(latest_path)
    return latest_paths



# def create_directory(path):
#     try:
#         if not os.path.exists(path):
#             os.makedirs(path)
#     except Exception as e:
#         print(f"创建目录时出错: {e}")

# def copy_files(file_paths, output_path):
#     for path in file_paths:
#         try:
#             curves = path.split("/")[4].split("_")[1]
#             destination_directory = os.path.join(output_path, key1, key2, curves)
#             create_directory(destination_directory)
#             shutil.copy(path, destination_directory)
#         except Exception as e:
#             print(f"复制文件时出错: {e}")

# 参数配置
directory_path = '/home/data/panghu/new/LibSignal/data/output_data/tsc'
file_extension = 'BRF.log'
# output_path = '../output_data/'
# 查找文件并进行分类过滤
file_paths = find_files(directory_path, file_extension)



# 排序关键字
keywords = ["fixedtime", "sotl", "maxpressure", "mplight", "frap", "myfrap"]
# 排序文件路径
sorted_file_paths = sort_files(file_paths, keywords)
# 打印排序后的文件路径
print("排序后的文件路径列表:")
for file_path in sorted_file_paths:
    print(file_path)

includes = [
    ['cityflow1x1','cityflow_mymplight'],
    ['cityflow1x1','cityflow_mplight'],
    # ['cityflow1x1','cityflow_frap'],
    ['cityflow1x1','cityflow_myfrap']
]

latest_paths = get_latest_paths(sorted_file_paths, includes)

print("最终筛选出的最新路径列表为:")
print(latest_paths)

# 对比mymplight和mplight、myfrap


# 从.log文件中读取数据

# 指定宋体字体
font = FontProperties(fname="/home/data/panghu/new/LibSignal/tnwsimsun.ttf", size=14)

print("11111111111111111111111111111111111111111111111111111111111111111111111111")
file_paths = latest_paths

def extract_parameter_from_path(file_path):
    # 获取路径的目录部分
    dir_path = os.path.dirname(file_path)
    # 分割目录路径
    parts = dir_path.split('/')
    # 提取第三部分（对应myyfrap或frap）
    parameter = parts[3].replace('cityflow_', '')
    return parameter
    # 从文件路径中获取myyfrap和frap参数

# 从文件路径中获取mymplight和mplight参数
names = []
for path in file_paths:
    names.append(extract_parameter_from_path(path))
print(names)


# 定义结构体类
class TrafficData:
    def __init__(self, name):
        self.name = name
        self.episode_list = []
        self.queue_list = []
        self.delay_list = []
        self.throughput_list = []
        self.travel_time_list = []


# 读取和解析数据
def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = TrafficData(extract_parameter_from_path(file_path))

    for i in range(len(lines)):
        line = lines[i]
        if line.startswith('episode:'):
            episode = int(line.split(':')[1].split('/')[0].strip())
            travel_time = float(line.split('real avg travel time:')[1].strip())
            data.travel_time_list.append(travel_time)

            # 读取前一行数据
            previous_line = lines[i - 1]
            queue = float(previous_line.split('queue:')[1].split(',')[0].strip())
            delay = float(previous_line.split('delay:')[1].split(',')[0].strip())
            throughput = float(previous_line.split('throughput:')[1].strip())

            data.episode_list.append(episode)
            data.queue_list.append(queue)
            data.delay_list.append(delay)
            data.throughput_list.append(throughput)

    return data


# 解析两个日志文件的数据
mymplight_data = parse_log_file(latest_paths[0])
mplight_data = parse_log_file(latest_paths[1])
myfrap_data = parse_log_file(latest_paths[2])


# 创建一个图形对象和四个子图
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 绘制Queue对比图
axs[0, 0].plot(mymplight_data.episode_list, mymplight_data.queue_list, label='mymplight_data')
axs[0, 0].plot(mplight_data.episode_list, mplight_data.queue_list, label='mplight_data', linestyle='--')
# axs[0, 0].plot(frap_data.episode_list, frap_data.queue_list, label='frap_data', linestyle='-.')
axs[0, 0].plot(myfrap_data.episode_list, myfrap_data.queue_list, label='myfrap_data', linestyle='--',color='red', linewidth=2)
axs[0, 0].set_xlabel('Episode')
axs[0, 0].set_ylabel('Queue')
axs[0, 0].legend()
axs[0, 0].set_title('Queue over Time')
axs[0, 0].grid(True)

# 绘制Delay对比图
axs[0, 1].plot(mymplight_data.episode_list, mymplight_data.delay_list, label='mymplight_data')
axs[0, 1].plot(mplight_data.episode_list, mplight_data.delay_list, label='mplight_data', linestyle='--')
# axs[0, 1].plot(frap_data.episode_list, frap_data.delay_list, label='frap_data', linestyle='-.')
axs[0, 1].plot(myfrap_data.episode_list, myfrap_data.delay_list, label='myfrap_data', linestyle='--',color='red', linewidth=2)
axs[0, 1].set_xlabel('Episode')
axs[0, 1].set_ylabel('Delay')
axs[0, 1].legend()
axs[0, 1].set_title('Delay over Time')
axs[0, 1].grid(True)

# 绘制Throughput对比图
axs[1, 0].plot(mymplight_data.episode_list, mymplight_data.throughput_list, label='mymplight_data')
axs[1, 0].plot(mplight_data.episode_list, mplight_data.throughput_list, label='mplight_data', linestyle='--')
# axs[1, 0].plot(frap_data.episode_list, frap_data.throughput_list, label='frap_data', linestyle='-.')
axs[1, 0].plot(myfrap_data.episode_list, myfrap_data.throughput_list, label='myfrap_data', linestyle='--',color='red', linewidth=2)
axs[1, 0].set_xlabel('Episode')
axs[1, 0].set_ylabel('Throughput')
axs[1, 0].legend()
axs[1, 0].set_title('Throughput over Time')
axs[1, 0].grid(True)

# 绘制Average Travel Time对比图
axs[1, 1].plot(range(len(mymplight_data.travel_time_list)), mymplight_data.travel_time_list, label='mymplight_data')
axs[1, 1].plot(range(len(mplight_data.travel_time_list)), mplight_data.travel_time_list, label='mplight_data', linestyle='--')
# axs[1, 1].plot(frap_data.episode_list, frap_data.travel_time_list, label='frap_data', linestyle='-.')
axs[1, 1].plot(myfrap_data.episode_list, myfrap_data.travel_time_list, label='myfrap_data', linestyle='--',color='red', linewidth=2)
axs[1, 1].set_xlabel('Episode')
axs[1, 1].set_ylabel('Average Travel Time')
axs[1, 1].legend()
axs[1, 1].set_title('Average Travel Time over Time')
axs[1, 1].grid(True)

# 调整子图布局
plt.tight_layout()

# 检查并创建目录
output_dir = './testoutput'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存图片到指定路径
output_path = os.path.join(output_dir, 'filter-compare_method.png')
plt.savefig(output_path, dpi=300)
plt.show()




















# rl_1x1 = filter_files(file_paths, "cityflow1x1/runep_rl/", 'colight')
# no_rl1x1 = filter_files(file_paths, "cityflow1x1/runep_no_rl/")
# rl_4x4 = filter_files(file_paths, "cityflow4x4/runep_rl/")
# no_rl4x4 = filter_files(file_paths, "cityflow4x4/runep_no_rl/")

# # 数据字典映射
# data_mapping = {
#     '1x1': {'rl_methods': rl_1x1, 'no_rl_methods': no_rl1x1},
#     '4x4': {'rl_methods': rl_4x4, 'no_rl_methods': no_rl4x4}
# }

# # 复制文件
# for key1, value1 in data_mapping.items():
#     for key2, file_paths in value1.items():
#         copy_files(file_paths, output_path)

# print("文件已按照目录结构复制完成。")
