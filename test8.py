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

# 从.log文件中读取数据

# 指定宋体字体
font = FontProperties(fname="/home/panghu/rl_tsc/Libsignal/tnwsimsun.ttf", size=14)
# 针对config4来对比
# new/LibSignal/data/output_data/tsc/cityflow_mplight/cityflow1x1_config4/rl_agent/logger/2024_07_27-11_44_02_BRF.log
# new/LibSignal/data/output_data/tsc/cityflow_colight/cityflow1x1_config4/rl_agent/logger/2024_07_27-11_44_02_BRF.log
# new/LibSignal/data/output_data/tsc/cityflow_myyfrap/cityflow1x1_config4/rl_agent/logger/2024_07_27-11_44_03_BRF.log
# 对比mplight、colight、myyfrap


file_path1 = 'data/output_data/tsc/cityflow_mplight/cityflow1x1_config5/rl_agent/logger/2024_07_27-11_44_03_BRF.log'
file_path2 = 'data/output_data/tsc/cityflow_colight/cityflow1x1_config5/rl_agent/logger/2024_07_27-11_44_03_BRF.log'
file_path3 = 'data/output_data/tsc/cityflow_myyfrap/cityflow1x1_config5/rl_agent/logger/2024_07_27-11_44_02_BRF.log'




file_paths = [file_path1, file_path2,file_path3]

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
mplight_data = parse_log_file(file_path1)
colight_data = parse_log_file(file_path2)
myfrap_data = parse_log_file(file_path3)


# 创建一个图形对象和四个子图
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 绘制Queue对比图
axs[0, 0].plot(colight_data.episode_list, colight_data.queue_list, label='colight_data')
axs[0, 0].plot(mplight_data.episode_list, mplight_data.queue_list, label='mplight_data', linestyle='--')
# axs[0, 0].plot(frap_data.episode_list, frap_data.queue_list, label='frap_data', linestyle='-.')
axs[0, 0].plot(myfrap_data.episode_list, myfrap_data.queue_list, label='myfrap_data', linestyle='--',color='red', linewidth=2)
axs[0, 0].set_xlabel('Episode')
axs[0, 0].set_ylabel('Queue')
axs[0, 0].legend()
axs[0, 0].set_title('Queue over Time')
axs[0, 0].grid(True)

# 绘制Delay对比图
axs[0, 1].plot(colight_data.episode_list, colight_data.delay_list, label='colight_data')
axs[0, 1].plot(mplight_data.episode_list, mplight_data.delay_list, label='mplight_data', linestyle='--')
# axs[0, 1].plot(frap_data.episode_list, frap_data.delay_list, label='frap_data', linestyle='-.')
axs[0, 1].plot(myfrap_data.episode_list, myfrap_data.delay_list, label='myfrap_data', linestyle='--',color='red', linewidth=2)
axs[0, 1].set_xlabel('Episode')
axs[0, 1].set_ylabel('Delay')
axs[0, 1].legend()
axs[0, 1].set_title('Delay over Time')
axs[0, 1].grid(True)

# 绘制Throughput对比图
axs[1, 0].plot(colight_data.episode_list, colight_data.throughput_list, label='colight_data')
axs[1, 0].plot(mplight_data.episode_list, mplight_data.throughput_list, label='mplight_data', linestyle='--')
# axs[1, 0].plot(frap_data.episode_list, frap_data.throughput_list, label='frap_data', linestyle='-.')
axs[1, 0].plot(myfrap_data.episode_list, myfrap_data.throughput_list, label='myfrap_data', linestyle='--',color='red', linewidth=2)
axs[1, 0].set_xlabel('Episode')
axs[1, 0].set_ylabel('Throughput')
axs[1, 0].legend()
axs[1, 0].set_title('Throughput over Time')
axs[1, 0].grid(True)

# 绘制Average Travel Time对比图
axs[1, 1].plot(colight_data.episode_list, colight_data.travel_time_list, label='colight_data')
axs[1, 1].plot(mplight_data.episode_list, mplight_data.travel_time_list, label='mplight_data', linestyle='--')
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
output_path = os.path.join(output_dir, 'test8.png')
plt.savefig(output_path, dpi=300)
plt.show()



