import argparse
import re
import numpy as np

def parse_log(log_file, max_round):
    """ 解析日志文件，提取轮数和准确率 """
    rounds = []
    accuracies = []

    round_pattern = re.compile(r"\*\*\*\* Round (\d+)/\d+ \*\*\*\*")
    accuracy_pattern = re.compile(r"Average accuracy: (\d+\.\d+)%")

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            round_match = round_pattern.search(line)
            accuracy_match = accuracy_pattern.search(line)

            if round_match:
                rounds.append(int(round_match.group(1)))

            if accuracy_match:
                accuracies.append(float(accuracy_match.group(1)))

            if len(rounds) >= max_round and len(accuracies) == len(rounds):
                break

    return rounds, accuracies

def process_logs(log_files, output_file, max_round):
    """ 处理多个日志文件并计算平均值，保存数据到文件 """
    all_rounds = []
    all_accuracies = []

    for log_file in log_files:
        log_file = "logs/" + log_file
        rounds, accuracies = parse_log(log_file, max_round)
        all_rounds.append(rounds)
        all_accuracies.append(accuracies)

    # 假设所有文件的轮数一致
    common_rounds = all_rounds[0]
    avg_accuracies = np.mean(np.array(all_accuracies), axis=0)

    with open(output_file, "w", encoding="utf-8") as f:
        for round_num, acc in zip(common_rounds, avg_accuracies):
            f.write(f"{round_num},{acc:.2f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process log files to compute average accuracy.")
    parser.add_argument('-f', '--files', type=str, nargs='+', required=True, help='List of log file paths to process.')
    parser.add_argument('-o', '--output', type=str, default='processed_data.csv', help='Output data filename.')
    parser.add_argument('-r', '--rounds', type=int, default=50, help='Maximum rounds.')

    args = parser.parse_args()
    process_logs(args.files, args.output, args.rounds)