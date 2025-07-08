import numpy as np
import os

n_bits_w = 2
n_bits_a = 6

# 原始的.npz文件路径
original_npz_path = 'experiments_log/w{}a{}_50k_samples.npz'.format(n_bits_w, n_bits_a)
# 加载原始数据
with np.load(original_npz_path) as data:
    all_samples = data['arr_0']  # 假设数据存储在'arr_0'键下

# 每个类别的样本数
samples_per_class = 50
# 要提取的样本数
num_samples_to_extract = 5

# 计算总共需要的样本数
total_samples = num_samples_to_extract * 1000

# 初始化一个空数组来存储提取的样本
extracted_samples = np.zeros((total_samples, *all_samples.shape[1:]), dtype=all_samples.dtype)

# 遍历每个类别并随机提取样本
class_index = 0
for class_start in range(0, len(all_samples), samples_per_class):
    # 从当前类别中随机选择10个样本
    selected_indices = np.random.choice(samples_per_class, num_samples_to_extract, replace=False)
    selected_samples = all_samples[class_start:class_start + samples_per_class][selected_indices]
    
    # 将选中的样本添加到提取的样本数组中
    extracted_samples[class_index * num_samples_to_extract:(class_index + 1) * num_samples_to_extract] = selected_samples
    
    # 更新类别索引
    class_index += 1

# 保存提取的样本到新的.npz文件
output_path = 'experiments_log/w{}a{}_5k_samples.npz'.format(n_bits_w, n_bits_a)
np.savez(output_path, extracted_samples)

print(f'Extracted samples saved to {output_path}')