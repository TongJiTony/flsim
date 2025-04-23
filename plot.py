import argparse
import re
import matplotlib.pyplot as plt

def parse_log(log_file):
    """ 解析日志文件，提取轮数和准确率 """
    rounds = []
    accuracies = []

    round_pattern = re.compile(r"\*\*\*\* Round (\d+)/30 \*\*\*\*")
    accuracy_pattern = re.compile(r"Average accuracy: (\d+\.\d+)%")

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            round_match = round_pattern.search(line)
            accuracy_match = accuracy_pattern.search(line)

            if round_match:
                rounds.append(int(round_match.group(1)))

            if accuracy_match:
                accuracies.append(float(accuracy_match.group(1)))

    return rounds, accuracies

def plot_accuracy(log_files, output_image):
    """ 绘制多个日志文件的对比图 """
    plt.figure(figsize=(20, 10))

    for log_file in log_files:
        rounds, accuracies = parse_log(log_file)
        label = log_file.split('/')[-1]  # 使用文件名作为标签
        plt.plot(rounds, accuracies, marker='o', linestyle='-', label=label)

    plt.xlabel("Round Number")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Trend Comparison Across Logs")
    plt.legend(loc = "upper right")
    plt.grid(True)

    # 保存图像
    plt.savefig(output_image)
    plt.show()

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Plot accuracy trend from multiple log files.")
    parser.add_argument('-f', '--files', type=str, nargs='+', required=True,
                        help='List of log file paths to process.')
    parser.add_argument('-o', '--output', type=str, default='accuracy_comparison.png',
                        help='Output image filename.')

    args = parser.parse_args()

    # 运行绘图函数
    plot_accuracy(args.files, args.output)