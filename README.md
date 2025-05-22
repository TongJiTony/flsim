# FLSim-DataInjection

本项目为基于FLSim联邦学习模拟框架搭建的用于研究数据注入对于联邦学习性能影响的模拟实验项目，在原有框架基础上，进行了以下内容的改进与更新。

### Fashion-MNIST数据集基准模型调参，fine-tuning
1. 调整卷积神经网络模型结构，提高准确率
2. 调整学习率，减少过拟合程度

### 三种数据注入方式的代码实现，详情可见dataInjection.py
1. 标签翻转注入（随机修改样本标签）
2. 虚假样本注入（向样本图像数据添加噪声）
3. 混合方式注入（随机采用前两种方式）
4. 数据注入的执行位于`client.py`的 `set_data()`中，即在训练开始前每一个客户端分配数据集的时候进行数据注入。

### 数据可视化工具集
1. `process_logs.py` 提炼多个日志数据中的关键评估指标（准确率或损失），计算平均值后生成csv文件储存。
2. `plot_data.py` 根据csv文件数据绘制图表，提供平滑曲线功能选项。
3. 两个文件的参数设置可输入 '--help' 查看用例![image](https://github.com/user-attachments/assets/5c7b7b68-c407-4f24-be6a-da04a60ef9ef)


### 其他改进
1. 修改`run.py`，使其支持 '-f' 选项，选择日志的输出位置（filename）
2. 修改`config.py`，使其支持与数据相关的配置选项：'injection'（是否注入数据）,'injection_method'（数据注入方法）
3. 开发`plot.py`，支持直接将日志数据转化为可视化图表，无需生成csv文件的中间步骤。具体参数设置与数据可视化工具集类似。

## About

Welcome to **FLSim**, a PyTorch based federated learning simulation framework, created for experimental research in a paper accepted by [IEEE INFOCOM 2020](https://infocom2020.ieee-infocom.org):

[Hao Wang](https://www.haow.ca), Zakhary Kaplan, [Di Niu](https://sites.ualberta.ca/~dniu/Homepage/Home.html), [Baochun Li](http://iqua.ece.toronto.edu/bli/index.html). "Optimizing Federated Learning on Non-IID Data with Reinforcement Learning," in the Proceedings of IEEE INFOCOM, Beijing, China, April 27-30, 2020.



## Installation

To install **FLSim**, all that needs to be done is clone this repository to the desired directory.

### Dependencies

**FLSim** uses [Anaconda](https://www.anaconda.com/distribution/) to manage Python and it's dependencies, listed in [`environment.yml`](environment.yml). To install the `fl-py37` Python environment, set up Anaconda (or Miniconda), then download the environment dependencies with:

```shell
conda env create -f environment.yml
```

## Usage

Before using the repository, make sure to activate the `fl-py37` environment with:

```shell
conda activate fl-py37
```

### Simulation

To start a simulation, run [`run.py`](run.py) from the repository's root directory:

```shell
python run.py
  --config=config.json
  --log=INFO
```

##### `run.py` flags

* `--config` (`-c`): path to the configuration file to be used.
* `--log` (`-l`): level of logging info to be written to console, defaults to `INFO`.

##### `config.json` files

**FLSim** uses a JSON file to manage the configuration parameters for a federated learning simulation. Provided in the repository is a generic template and three preconfigured simulation files for the CIFAR-10, FashionMNIST, and MNIST datasets.

For a detailed list of configuration options, see the [wiki page](https://github.com/iQua/flsim/wiki/Configuration).

If you have any questions, please feel free to contact Hao Wang (haowang@ece.utoronto.ca)
