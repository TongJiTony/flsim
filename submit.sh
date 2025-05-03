#!/bin/bash 



#SBATCH --job-name=myFisrtJob            # 作业名
#SBATCH --comment="just hello world "    # 作业描述

#SBATCH --partition=intel        # 使用哪个分区

#SBATCH --output=%x_%j.out       # 输出文件
#SBATCH --error=%x_%j.err        # 错误输出文件

#SBATCH --time=0-00:05:00        # 时间限制5分钟
#SBATCH --nodes=1                # 申请1个节点
#SBATCH --ntasks=2               # 申请2个任务(进程)
#SBATCH --cpus-per-task=1        # 每个任务用1个cpu
#SBATCH --mem-per-cpu=10g        # 每个cpu用10G内存

echo 'hello world '              # 开始你的表演，编写或调用你自己的代码。

# 调用您自己的程序.....
python run.py
