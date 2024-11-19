import time
import re
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.font_manager import FontProperties
import json
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import csv
import pandas as pd

# 打印一下工作路径
print("Current Working Directory:", os.getcwd())
# 转换一下工作目录
os.chdir('/home/data/panghu/new/LibSignal')
print("Current Working Directory:", os.getcwd())

# 从.log文件中读取数据

# log文件如下

# 指定宋体字体
font = FontProperties(fname="/home/panghu/rl_tsc/Libsignal/tnwsimsun.ttf", size=14)
# rl

file_path1 = 'data/output_data/tsc/cityflow_colight/cityflow1x1/test/logger/2024_05_29-23_19_42_BRF.log'
file_path2 = 'data/output_data/tsc/cityflow_dqn/cityflow1x1/test/logger/2024_05_29-21_58_23_BRF.log'
file_path3 = 'data/output_data/tsc/cityflow_frap/cityflow1x1/test/logger/2024_05_29-22_33_29_BRF.log'
file_path4 = 'data/output_data/tsc/cityflow_mplight/cityflow1x1_config1/real_delay/logger/2024_06_27-23_46_08_BRF.log'
# no_rl
file_path5 = 'data/output_data/tsc/cityflow_fixedtime/cityflow1x1/test/logger/2024_05_29-22_33_01_BRF.log'
file_path6 = 'data/output_data/tsc/cityflow_maxpressure/cityflow1x1/test/logger/2024_05_30-13_35_33_BRF.log'
file_path7 = 'data/output_data/tsc/cityflow_sotl/cityflow1x1/test/logger/2024_06_09-22_26_50_BRF.log'


rl_file_paths = [file_path1, file_path2,file_path3,file_path4]
no_rl_file_paths = [file_path5, file_path6,file_path7]

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
for path in rl_file_paths:
    names.append(extract_parameter_from_path(path))
print(names)

for path in no_rl_file_paths:
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


# 读取和解析rl数据
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


# 读取解析no_rl数据
def parse_log_no_rl_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    name = extract_parameter_from_path(file_path)
    data = TrafficData(name)

    # 假设数据在文件的最后两行
    for line in lines:
        if "Final Travel Time" in line:
            travel_time = float(line.split('Final Travel Time is ')[1].split(',')[0].strip())
            reward = float(line.split('mean rewards:')[1].split(',')[0].strip())
            queue = float(line.split('queue:')[1].split(',')[0].strip())
            delay = float(line.split('delay:')[1].split(',')[0].strip())
            throughput = float(line.split('throughput:')[1].strip())

            # 将数据添加到TrafficData实例中
            data.travel_time_list.append(travel_time)
            # data.reward.append(reward)
            data.queue_list.append(queue)
            data.delay_list.append(delay)
            data.throughput_list.append(throughput)

        elif "Total time taken" in line:
            total_time = float(line.split('Total time taken: ')[1].strip())
            # data.total_time.append(total_time)

    # 将 episodes 列表填充为 0 到 199
    total_records = 200
    data.episode_list = list(range(total_records))

    # 填充其他列表为固定值
    data.delay_list = [data.delay_list[0]] * total_records if data.delay_list else [0] * total_records
    data.queue_list = [data.queue_list[0]] * total_records if data.queue_list else [0] * total_records  # 假设队列固定为第一个值
    data.throughput_list = [data.throughput_list[0]] * total_records if data.throughput_list else [0] * total_records  # 假设吞吐量固定为第一个值
    data.travel_time_list = [data.travel_time_list[0]] * total_records if data.travel_time_list else [0] * total_records  # 假设旅行时间固定为第一个值
    # data.reward = [data.reward[0]] * total_records if data.reward else [0] * total_records  # 假设奖励固定为第一个值
    # data.total_time = [data.total_time[0]] * total_records if data.total_time else [0] * total_records  # 假设总时间固定为第一个值

    return data






# 解析数据
print(file_path1)

if not os.path.exists(file_path1):
    print(f"Error: File {file_path1} does not exist.")

colight_data = parse_log_file(file_path1)
dqn_data = parse_log_file(file_path2)
frap_data = parse_log_file(file_path3)
mplight_data = parse_log_file(file_path4)


fixedtime_data = parse_log_no_rl_file(file_path5)
maxpressure_data = parse_log_no_rl_file(file_path6)
sotl_data = parse_log_no_rl_file(file_path7)


# # 创建一个图形对象
# plt.figure(figsize=(10, 6))

# # 绘制数据
# plt.plot(dqn_data.episode_list, dqn_data.travel_time_list, label='dqn_data', linestyle='--')
# plt.plot(fixedtime_data.episode_list, fixedtime_data.travel_time_list, label='fixedtime_data', linestyle='-.')
# plt.plot(colight_data.episode_list, colight_data.travel_time_list, label='colight_data', linestyle='--')
# plt.plot(myyfrap_data.episode_list, myyfrap_data.travel_time_list, label='myyfrap_data', linestyle=':')


# # 设置标签和标题
# plt.xlabel('Episode')
# plt.ylabel('travel_time')
# plt.title('travel over Time')
# plt.legend()
# plt.grid(True)


# # 检查并创建目录
# output_dir = './testoutput'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # 保存图片到指定路径
# output_path = os.path.join(output_dir, '1x6-1.png')
# plt.savefig(output_path, dpi=300)
# plt.show()







# # 查看数据
# # 打印fixedtime_data的数据
# print("Fixed Time Data:")
# print("Episodes:", fixedtime_data.episode_list)
# print("Queues:", fixedtime_data.queue_list)
# print("Delays:", fixedtime_data.delay_list)
# print("Throughputs:", fixedtime_data.throughput_list)
# print("Travel Times:", fixedtime_data.travel_time_list)

# # 打印myyfrap_data的数据
# print("\nMyyfrap Data:")
# print("Episodes:", myyfrap_data.episode_list)
# print("Queues:", myyfrap_data.queue_list)
# print("Delays:", myyfrap_data.delay_list)
# print("Throughputs:", myyfrap_data.throughput_list)
# print("Travel Times:", myyfrap_data.travel_time_list)



# 创建一个图形对象和四个子图
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 绘制Queue对比图
axs[0, 0].plot(colight_data.episode_list, colight_data.queue_list, label='colight_data')
axs[0, 0].plot(dqn_data.episode_list, dqn_data.queue_list, label='dqn_data', linestyle='--')
axs[0, 0].plot(fixedtime_data.episode_list, fixedtime_data.queue_list, label='fixedtime_data', linestyle='-.')
axs[0, 0].plot(frap_data.episode_list, frap_data.queue_list, label='frap_data', linestyle='--',color='red', linewidth=2)
axs[0, 0].plot(mplight_data.episode_list, mplight_data.queue_list, label='mplight_data')
axs[0, 0].plot(maxpressure_data.episode_list, maxpressure_data.queue_list, label='maxpressure_data')
axs[0, 0].plot(sotl_data.episode_list, sotl_data.queue_list, label='sotl_data')
axs[0, 0].set_xlabel('Episode')
axs[0, 0].set_ylabel('Queue')
axs[0, 0].legend()
axs[0, 0].set_title('Queue over Time')
axs[0, 0].grid(True)

# 绘制Delay对比图
axs[0, 1].plot(colight_data.episode_list, colight_data.delay_list, label='colight_data')
axs[0, 1].plot(dqn_data.episode_list, dqn_data.delay_list, label='dqn_data', linestyle='--')
axs[0, 1].plot(fixedtime_data.episode_list, fixedtime_data.delay_list, label='fixedtime_data', linestyle='-.')
axs[0, 1].plot(frap_data.episode_list, frap_data.delay_list, label='frap_data', linestyle='--',color='red', linewidth=2)
axs[0, 1].plot(mplight_data.episode_list, mplight_data.delay_list, label='mplight_data')
axs[0, 1].plot(maxpressure_data.episode_list, maxpressure_data.delay_list, label='maxpressure_data')
axs[0, 1].plot(sotl_data.episode_list, sotl_data.delay_list, label='sotl_data')
axs[0, 1].set_xlabel('Episode')
axs[0, 1].set_ylabel('Delay')
axs[0, 1].legend()
axs[0, 1].set_title('Delay over Time')
axs[0, 1].grid(True)

# 绘制Throughput对比图
axs[1, 0].plot(colight_data.episode_list, colight_data.throughput_list, label='colight_data')
axs[1, 0].plot(dqn_data.episode_list, dqn_data.throughput_list, label='dqn_data', linestyle='--')
axs[1, 0].plot(fixedtime_data.episode_list, fixedtime_data.throughput_list, label='fixedtime_data', linestyle='-.')
axs[1, 0].plot(frap_data.episode_list, frap_data.throughput_list, label='frap_data', linestyle='--',color='red', linewidth=2)
axs[1, 0].plot(mplight_data.episode_list, mplight_data.throughput_list, label='mplight_data')
axs[1, 0].plot(maxpressure_data.episode_list, maxpressure_data.throughput_list, label='maxpressure_data')
axs[1, 0].plot(sotl_data.episode_list, sotl_data.throughput_list, label='sotl_data')
axs[1, 0].set_xlabel('Episode')
axs[1, 0].set_ylabel('Throughput')
axs[1, 0].legend()
axs[1, 0].set_title('Throughput over Time')
axs[1, 0].grid(True)

# 绘制Average Travel Time对比图
axs[1, 1].plot(colight_data.episode_list, colight_data.travel_time_list, label='colight_data')
axs[1, 1].plot(dqn_data.episode_list, dqn_data.travel_time_list, label='dqn_data', linestyle='--')
axs[1, 1].plot(fixedtime_data.episode_list, fixedtime_data.travel_time_list, label='fixedtime_data', linestyle='-.')
axs[1, 1].plot(frap_data.episode_list, frap_data.travel_time_list, label='frap_data', linestyle='--',color='red', linewidth=2)
axs[1, 1].plot(mplight_data.episode_list, mplight_data.travel_time_list, label='mplight_data')
axs[1, 1].plot(maxpressure_data.episode_list, maxpressure_data.travel_time_list, label='maxpressure_data')
axs[1, 1].plot(sotl_data.episode_list, sotl_data.travel_time_list, label='sotl_data')
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
output_path = os.path.join(output_dir, '1x1-4.png')
plt.savefig(output_path, dpi=300)
plt.show()



