import logging
import torch
import torch.nn as nn
import torch.optim as optim

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

def label_flip(sample, original_label, new_label):
    data, label = sample
    if label == original_label:
        label = new_label

    return (data, label)

def data_injection(data, percentage):
    for i in range(0, int(len(data)/10)):
        sample = data[10*i]
        _, original_label = sample
        # produce new label in 1 or 2, while new label will not be the same with original one
        new_label = (original_label+1) % 10
        data[i] = label_flip(sample, original_label, new_label)
        logging.info('Pick #{} sample, flipping label {} to label {}'.format(10*i, original_label, new_label))
    return data
