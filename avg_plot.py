import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def smooth_data(y_values, window_size=23, poly_order=3):
    return savgol_filter(y_values, window_size, poly_order)

def parse_log(log_file, max_round, loss_curve):
    """ 解析日志文件，提取轮数和准确率 """
    rounds = []
    accuracies = []
    loss = []

    round_pattern = re.compile(r"\*\*\*\* Round (\d+)/\d+ \*\*\*\*")
    accuracy_pattern = re.compile(r"Average accuracy: (\d+\.\d+)%")
    loss_pattern = re.compile(r"Loss: (\d+\.\d+)")

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            round_match = round_pattern.search(line)
            accuracy_match = accuracy_pattern.search(line)
            loss_match = loss_pattern.search(line)

            if round_match:
                rounds.append(int(round_match.group(1)))

            if loss_match and loss_curve:
                loss.append(float(loss_match.group(1)))

            if accuracy_match:
                accuracies.append(float(accuracy_match.group(1)))

            if len(rounds) >= max_round and len(accuracies) == len(rounds):
                break

    return rounds, accuracies, loss

def plot_comparison(log_files_set1, log_files_set2, output_image, max_round, loss_curve_flag):
    """ 绘制两组日志文件的平均曲线进行比较 """
    plt.figure(figsize=(15, 10))
    ax1 = plt.gca()  # 主 y 轴用于准确率
    if loss_curve_flag:
        ax2 = ax1.twinx()

    def compute_average(log_files):
        all_rounds = []
        all_accuracies = []
        all_losses = []
        
        for log_file in log_files:
            log_file = "logs/" + log_file
            rounds, accuracies, losses = parse_log(log_file, max_round, loss_curve_flag)
            all_rounds.append(rounds)
            all_accuracies.append(accuracies)
            all_losses.append(losses)

        common_rounds = all_rounds[0]
        avg_accuracies = np.mean(np.array(all_accuracies), axis=0)
        avg_losses = np.mean(np.array(all_losses), axis=0)

        return common_rounds, avg_accuracies, avg_losses

    rounds1, avg_accuracies1, avg_losses1 = compute_average(log_files_set1)
    rounds2, avg_accuracies2, avg_losses2 = compute_average(log_files_set2)

    smoothed_avg_accuracies1 = smooth_data(avg_accuracies1)
    smoothed_avg_accuracies2 = smooth_data(avg_accuracies2)

    ax1.plot(rounds1, smoothed_avg_accuracies1, linestyle='-', color='blue', label="Average Accuracy - Set 1")
    ax1.plot(rounds2, smoothed_avg_accuracies2, linestyle='-', color='green', label="Average Accuracy - Set 2")

    if loss_curve_flag:
        smoothed_avg_losses1 = smooth_data(avg_losses1)
        smoothed_avg_losses2 = smooth_data(avg_losses2)

        ax2.plot(rounds1, smoothed_avg_losses1, linestyle='--', color='red', label="Average Loss - Set 1")
        ax2.plot(rounds2, smoothed_avg_losses2, linestyle='--', color='orange', label="Average Loss - Set 2")

    ax1.set_xlabel("Round Number")
    ax1.set_ylabel("Accuracy (%)", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title("Comparison of Two Average Accuracy & Loss Curves")
    ax1.legend(loc="upper right", prop={'size': 14})
    
    if loss_curve_flag:
        ax2.set_ylabel("Loss", color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.legend(loc="upper left", prop={'size': 14})

    plt.grid(True)
    plt.savefig("charts/" + output_image)
    plt.show()

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Compare two sets of accuracy trend from log files.")
    parser.add_argument('-f1', '--files_set1', type=str, nargs='+', required=True, help='First set of log file paths.')
    parser.add_argument('-f2', '--files_set2', type=str, nargs='+', required=True, help='Second set of log file paths.')
    parser.add_argument('-o', '--output', type=str, default='comparison.png', help='Output image filename.')
    parser.add_argument('-r', '--rounds', type=int, default=50, help='Maximum rounds')
    parser.add_argument('-l', '--loss', action="store_true", help="Draw loss curve")

    args = parser.parse_args()
    
    # 运行比较函数
    plot_comparison(args.files_set1, args.files_set2, args.output, args.rounds, args.loss)