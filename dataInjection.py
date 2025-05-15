import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Data Injection: Label Flipping, Adversarial Injection, False Data Injection
    # Fashion-MINST graph data sample : (tensor([[[...]]]), 2)
    # label meanings:
    # 0	T-shirt/top
    # 1	Trouser
    # 2	Pullover
    # 3	Dress
    # 4	Coat
    # 5	Sandal
    # 6	Shirt
    # 7	Sneaker
    # 8	Bag
    # 9	Ankle boot

def label_flip(sample, num_classes=10):
    """
    对FashionMNIST数据进行标签翻转

    参数:
    sample (tuple): (image tensor, label)
    num_classes (int): 标签类别数量

    返回:
    tuple: (image tensor, 随机label)
    """
    data, label = sample
    false_label = label
    while false_label == label:
        false_label = np.random.randint(0, num_classes)

    return (data, false_label)

# Function to generate false sample by applying transformations
def false_sample(sample, mean=0, std=0.1):
    """
    给FashionMNIST数据添加高斯噪声

    参数:
    sample (tuple): (image tensor, label)
    mean (float): 高斯噪声的均值，整体偏移图片亮度
    std (float): 高斯噪声的标准差，影响图片噪声强度

    返回:
    tuple: (加噪后的image tensor, 原label)
    """
    data, label = sample
    noise = torch.randn_like(data) * std + mean
    noisy_data = data + noise
    noisy_data = torch.clamp(noisy_data, 0, 1)  # 确保像素值仍然在 [0,1] 之间
    return noisy_data, label

def data_injection(data, percentage, method):
    injection_option = method
    option_list = ["label_flip", "false_sample"]
    for i in range(0, int(len(data)*percentage)):
        sample = data[i]
        _, label = sample

        if method == "mixed":
            injection_option = np.random.choice(option_list)

        # label flipping
        if injection_option == "label_flip":
            data[i] = label_flip(sample)
            logging.debug('Pick #{} sample, flipping label {} to label {}'.format(i, label, data[i][1]))
        elif injection_option == "false_sample":
            data[i] = false_sample(sample)
            logging.debug('Pick #{} sample, creating false sample of label {}'.format(i, label))
            
    return data
