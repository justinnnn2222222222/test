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

# /home/data/panghu/new/LibSignal/data/output_data/tsc/cityflow_sotl/cityflow1x1_config1/real_delay/logger/2024_06_27-23_22_13_BRF.log

# 指定宋体字体
font = FontProperties(fname="/home/panghu/rl_tsc/Libsignal/tnwsimsun.ttf", size=14)


file_path1 = 'data/output_data/tsc/cityflow_sotl/cityflow1x1_config1/real_delay/logger/2024_06_27-23_22_13_BRF.log'
file_path2 = 'data/output_data/tsc/cityflow_fixedtime/cityflow1x6/test/logger/2024_10_26-06_50_45_BRF.log'

class TrafficData:
    def __init__(self, name):
        self.name = name
        self.episodes = []
        self.queue = []
        self.delay = []
        self.throughput = []
        self.travel_time = []
        self.reward = []
        self.total_time = []

    # def print_data(self):
    #     print(f"Data for {self.name}:")
    #     print("Episodes:", self.episodes)
    #     print("Queue:", self.queue)
    #     print("Delay:", self.delay)
    #     print("Throughput:", self.throughput)
    #     print("Travel Time:", self.travel_time)
    #     print("Reward:", self.reward)
    #     print("Total Time:", self.total_time)


# 从文件路径中获取name参数
def extract_parameter_from_path(file_path):
    # 获取路径的目录部分
    dir_path = os.path.dirname(file_path)
    # 分割目录路径
    parts = dir_path.split('/')
    # 提取第三部分（对应myyfrap或frap）
    parameter = parts[3].replace('cityflow_', '')   
    # 从文件路径中获取myyfrap和frap参数
    return parameter




def parse_log_file(file_path):
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
            data.travel_time.append(travel_time)
            data.reward.append(reward)
            data.queue.append(queue)
            data.delay.append(delay)
            data.throughput.append(throughput)

        elif "Total time taken" in line:
            total_time = float(line.split('Total time taken: ')[1].strip())
            data.total_time.append(total_time)

    # 将 episodes 列表填充为 0 到 199
    total_records = 200
    data.episodes = list(range(total_records))

    # 填充其他列表为固定值
    data.delay = [data.delay[0]] * total_records if data.queue else [0] * total_records
    data.queue = [data.queue[0]] * total_records if data.queue else [0] * total_records  # 假设队列固定为第一个值
    data.throughput = [data.throughput[0]] * total_records if data.throughput else [0] * total_records  # 假设吞吐量固定为第一个值
    data.travel_time = [data.travel_time[0]] * total_records if data.travel_time else [0] * total_records  # 假设旅行时间固定为第一个值
    data.reward = [data.reward[0]] * total_records if data.reward else [0] * total_records  # 假设奖励固定为第一个值
    data.total_time = [data.total_time[0]] * total_records if data.total_time else [0] * total_records  # 假设总时间固定为第一个值


    return data

# file_path1 = 'data/output_data/tsc/cityflow_sotl/cityflow1x1_config1/real_delay/logger/2024_06_27-23_22_13_BRF.log'
# file_path2 = 'data/output_data/tsc/cityflow_fixedtime/cityflow1x6/test/logger/2024_10_26-06_50_45_BRF.log'

# 使用示例
sotl_data = parse_log_file(file_path1)
fixedtime_data = parse_log_file(file_path2)



# # 创建一个图形对象和四个子图
# fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 创建一个图形对象
plt.figure(figsize=(10, 6))

# 绘制数据
plt.plot(sotl_data.episodes, sotl_data.queue, label='sotl_data', linestyle='--')
plt.plot(fixedtime_data.episodes, fixedtime_data.queue, label='fixedtime_data', linestyle='-.')


# 设置标签和标题
plt.xlabel('Episode')
plt.ylabel('Queue')
plt.title('Queue over Time')
plt.legend()
plt.grid(True)


# 检查并创建目录
output_dir = './testoutput'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存图片到指定路径
output_path = os.path.join(output_dir, 'ceshi_no_rl.png')
plt.savefig(output_path, dpi=300)
plt.show()

# print("name:", sotl_data.name)
# print("name:", fixedtime_data.name)

# 打印解析结果
# print("name:", parsed_data.name)
# print("episodes:", parsed_data.episodes)
# print("Reward:", parsed_data.reward)
# print("Queue:", parsed_data.queue)
# print("Delay:", parsed_data.delay)
# print("Throughput:", parsed_data.throughput)
# print("Travel Time:", parsed_data.travel_time)
# print("Total Time:", parsed_data.total_time)



