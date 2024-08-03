
# /home/data/panghu/new/LibSignal/data/output_data/tsc/cityflow_frap/cityflow1x1_config1/real_delay/logger/2024_06_27-23_46_08_BRF.log

# /home/data/panghu/new/LibSignal/data/output_data/tsc/cityflow_myfrap/cityflow1x1_config1/real_delay/logger/2024_06_27-23_46_07_BRF.log
#/home/data/panghu/new/LibSignal/data/output_data/tsc/cityflow_frap/cityflow1x1_config1/real_delay/logger/2024_06_27-23_46_08_BRF.log
import matplotlib.pyplot as plt

def parse_log_file(log_file_path):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    reward_list = []
    episode_list = []

    for i in range(len(lines)):
        line = lines[i]
        if line.startswith('episode:'):
            episode = int(line.split(':')[1].split('/')[0].strip())
            
            episode_list.append(episode)
            
        elif line.startswith('Test step:'):
            try:
                reward_part = line.split('rewards:')
                if len(reward_part) > 1:
                    reward = float(reward_part[1].split(',')[0].strip())
                    reward_list.append(reward)
            except ValueError as e:
                print(f"Error parsing reward value in line: {line}\n{e}")
        

    return episode_list, reward_list

# 解析第一个日志文件
log_file_path1 = '/home/data/panghu/new/LibSignal/data/output_data/tsc/cityflow_frap/cityflow1x1_config1/real_delay/logger/2024_06_27-23_46_08_BRF.log'
episodes1, reward1 = parse_log_file(log_file_path1)

# 解析第二个日志文件
log_file_path2 = '/home/data/panghu/new/LibSignal/data/output_data/tsc/cityflow_myfrap/cityflow1x1_config1/real_delay/logger/2024_06_27-23_46_07_BRF.log'
episodes2, reward2 = parse_log_file(log_file_path2)

# 只保留前50个数据点
episodes1 = episodes1[:50]
reward1 = reward1[:50]
episodes2 = episodes2[:50]
reward2 = reward2[:50]

# print(episodes1)
# print(episodes2)

# 绘制 reward 对比图
plt.plot(episodes1, reward1, label='frap', linestyle='-')
plt.plot(episodes2, reward2, label='myfrap', linestyle='--')

plt.title('Reward Comparison')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)

# 保存图片到文件
output_image_path = '/home/data/panghu/new/LibSignal/reward_comparison.png'
plt.savefig(output_image_path)

# 显示图像
plt.show()

