import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def smooth_data(y_values, window_size=13, poly_order=3):
    return savgol_filter(y_values, window_size, poly_order)

def read_data(input_file):
    """ 读取数据文件 """
    rounds = []
    accuracies = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(',')
            rounds.append(int(parts[0]))
            accuracies.append(float(parts[1]))

    return rounds, accuracies

def plot_data(input_files, output_image, loss_flag):
    """ 绘制多个数据文件的曲线 """
    plt.figure(figsize=(12, 6))
    
    for input_file in input_files:
        rounds, accuracies = read_data(input_file)
        accuracies = smooth_data(np.array(accuracies))
        plt.plot(rounds, accuracies, linewidth=4, linestyle='-', label=input_file.split('.')[0])  # 直接使用文件名作为标签

    plt.xlabel("Round Number", fontsize=18)
    plt.xticks(fontsize=14)
    if loss_flag:
        plt.ylabel("Loss", fontsize=18)
        plt.yticks(np.arange(0, 2, 0.2), fontsize=14)
        plt.legend(loc = "upper right", prop = {'size': 18})
    else:
        plt.ylabel("Accuracy(%)", fontsize=18)
        plt.legend(loc = "lower right", prop = {'size': 18})
    plt.grid(True)

    plt.savefig("charts/" + output_image)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot accuracy/loss trend from multiple processed data files.")
    parser.add_argument('-i', '--input', type=str, nargs='+', required=True, help='List of input data files.')
    parser.add_argument('-o', '--output', type=str, default='accuracy_plot.png', help='Output image filename.')
    parser.add_argument('-l',  '--loss', action='store_true', help="compute loss instead of accuracy")

    args = parser.parse_args()
    plot_data(args.input, args.output, args.loss)