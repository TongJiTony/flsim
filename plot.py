import argparse
import re
import matplotlib.pyplot as plt

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

def plot_accuracy(log_files, output_image, max_round, loss_curve_flag):
    """ 绘制多个日志文件的对比图 """
    plt.figure(figsize=(20, 10))
    ax1 = plt.gca()  # Primary y-axis for accuracy
    ax2 = ax1.twinx()

    for log_file in log_files:
        log_file = "logs/" + log_file
        rounds, accuracies, losses = parse_log(log_file, max_round, loss_curve_flag)
        label = log_file.split('/')[-1]  # 使用文件名作为标签
        ax1.plot(rounds, accuracies, marker='o', linestyle='-', label=label)
        if loss_curve_flag:
            ax2.plot(rounds, losses, marker='s', linestyle='--', color='r', label=f"Loss - {label}")

    ax1.set_xlabel("Round Number")
    ax1.set_ylabel("Accuracy (%)", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title("Accuracy & Loss Trend Comparison Across Logs")
    ax1.legend(loc = "upper right", prop = {'size': 14})
    ax2.set_ylabel("Loss", color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc = "upper left", prop = {'size': 14})
    plt.grid(True)

    # 保存图像
    plt.savefig("charts/" + output_image)
    plt.show()

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Plot accuracy trend from multiple log files.")
    parser.add_argument('-f', '--files', type=str, nargs='+', required=True,
                        help='List of log file paths to process.')
    parser.add_argument('-o', '--output', type=str, default='accuracy_comparison.png',
                        help='Output image filename.')
    parser.add_argument('-r', '--rounds', type=int, default=30,
                        help= 'Maximun rounds')
    parser.add_argument('-l', '--loss', action="store_true",
                        help="draw loss curve")

    args = parser.parse_args()

    # 运行绘图函数
    plot_accuracy(args.files, args.output, args.rounds, args.loss)