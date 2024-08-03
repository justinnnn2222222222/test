
# /home/data/panghu/new/LibSignal/data/output_data/tsc/cityflow_frap/cityflow1x1_config1/real_delay/logger/2024_06_27-23_46_08_BRF.log

# /home/data/panghu/new/LibSignal/data/output_data/tsc/cityflow_myfrap/cityflow1x1_config1/real_delay/logger/2024_06_27-23_46_07_BRF.log
#/home/data/panghu/new/LibSignal/data/output_data/tsc/cityflow_frap/cityflow1x1_config1/real_delay/logger/2024_06_27-23_46_08_BRF.log
import matplotlib.pyplot as plt

def parse_log_file(log_file_path):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    queue_list = []
    delay_list = []
    throughput_list = []
    travel_time_list = []
    episode_list = []

    for i in range(len(lines)):
        line = lines[i]
        if line.startswith('episode:'):
            episode = int(line.split(':')[1].split('/')[0].strip())
            travel_time = float(line.split('real avg travel time:')[1].strip())
            travel_time_list.append(travel_time)
            queue = float(lines[i-1].split('queue:')[1].split(',')[0].strip())
            delay = float(lines[i-1].split('delay:')[1].split(',')[0].strip())
            throughput = float(lines[i-1].split('throughput:')[1].strip())
            episode_list.append(episode)
            queue_list.append(queue)
            delay_list.append(delay)
            throughput_list.append(throughput)

    return episode_list, queue_list, delay_list, throughput_list, travel_time_list

# 解析第一个日志文件
log_file_path1 = '/home/data/panghu/new/LibSignal/data/output_data/tsc/cityflow_frap/cityflow1x1_config1/real_delay/logger/2024_06_27-23_46_08_BRF.log'
episodes1, queue_list1, delay_list1, throughput_list1, travel_time_list1 = parse_log_file(log_file_path1)

# 解析第二个日志文件
log_file_path2 = '/home/data/panghu/new/LibSignal/data/output_data/tsc/cityflow_myfrap/cityflow1x1_config1/real_delay/logger/2024_06_27-23_46_07_BRF.log'
episodes2, queue_list2, delay_list2, throughput_list2, travel_time_list2 = parse_log_file(log_file_path2)

# 创建4个子图
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 绘制queue对比图
axs[0, 0].plot(episodes1, queue_list1, label='frap', linestyle='-')
axs[0, 0].plot(episodes2, queue_list2, label='myfrap', linestyle='--')
axs[0, 0].set_title('Queue')
axs[0, 0].set_xlabel('Episode')
axs[0, 0].set_ylabel('Queue')
axs[0, 0].legend()
axs[0, 0].grid(True)

# 绘制delay对比图
axs[0, 1].plot(episodes1, delay_list1, label='frap', linestyle='-')
axs[0, 1].plot(episodes2, delay_list2, label='myfrap', linestyle='--')
axs[0, 1].set_title('Delay')
axs[0, 1].set_xlabel('Episode')
axs[0, 1].set_ylabel('Delay')
axs[0, 1].legend()
axs[0, 1].grid(True)

# 绘制throughput对比图
axs[1, 0].plot(episodes1, throughput_list1, label='frap', linestyle='-')
axs[1, 0].plot(episodes2, throughput_list2, label='myfrap', linestyle='--')
axs[1, 0].set_title('Throughput')
axs[1, 0].set_xlabel('Episode')
axs[1, 0].set_ylabel('Throughput')
axs[1, 0].legend()
axs[1, 0].grid(True)

# 绘制average travel time对比图
axs[1, 1].plot(range(len(travel_time_list1)), travel_time_list1, label='frap', linestyle='-')
axs[1, 1].plot(range(len(travel_time_list2)), travel_time_list2, label='myfrap', linestyle='--')
axs[1, 1].set_title('Average Travel Time')
axs[1, 1].set_xlabel('Episode')
axs[1, 1].set_ylabel('Average Travel Time')
axs[1, 1].legend()
axs[1, 1].grid(True)

# 调整子图布局
plt.tight_layout()

# 保存图片到文件
output_image_path = '/home/data/panghu/new/LibSignal/metrics_comparison_over_time.png'
plt.savefig(output_image_path)

# 显示图像
plt.show()