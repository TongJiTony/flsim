import load_data
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Training settings
lr = 0.001
momentum = 0.5
log_interval = 10

# Cuda settings
use_cuda = torch.cuda.is_available()
device = torch.device (  # pylint: disable=no-member
    'cuda' if use_cuda else 'cpu')


class Generator(load_data.Generator):
    """Generator for FashionMNIST dataset."""

    # Extract FashionMNIST data using torchvision datasets
    def read(self, path):
        self.trainset = datasets.FashionMNIST(
            path, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.testset = datasets.FashionMNIST(
            path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.labels = list(self.trainset.classes)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(5 * 5 * 64, 10)
    
    def forward(self, x):
        out = self.pool1(self.layer1(x))
        out = self.pool2(self.layer3(self.layer2(out)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=lr)


def get_trainloader(trainset, batch_size):
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


def get_testloader(testset, batch_size):
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


def extract_weights(model):
    weights = []
    for name, weight in model.to(torch.device('cpu')).named_parameters():  # pylint: disable=no-member
        if weight.requires_grad:
            weights.append((name, weight.data))

    return weights


def load_weights(model, weights):
    updated_state_dict = {}
    for name, weight in weights:
        updated_state_dict[name] = weight

    model.load_state_dict(updated_state_dict, strict=False)


def train(model, trainloader, optimizer, epochs):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):

        for batch_id, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
 
            loss.backward()
            optimizer.step()
            if batch_id % log_interval == 0:
                logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, epochs, loss.item()))


from sklearn.metrics import f1_score, recall_score

def test(model, testloader):
    model.to(device)
    # model.eval()
    model.train()

    all_labels = []
    all_predictions = []
    total_loss = 0.0  # 用于累计 loss
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # 计算 loss
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)  # 乘以 batch size 因为 loss 
            # logging.info('outputs:{}, labels:{}'.format(outputs,labels))
            # logging.info('single loss:{:.4f}'.format(loss.item()))
            
            predicted = torch.argmax(outputs, dim=1)  # 获取预测结果

            # 收集所有标签和预测结果
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # 计算准确率
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算平均 loss
    avg_loss = total_loss / total
    logging.debug('labels:{}, images:{}'.format(labels.size(0),images.size(0)))
    logging.debug('total_loss:{}, total:{}'.format(total_loss, total))
    logging.debug('Loss: {:.4f}'.format(avg_loss))

    # 计算准确率
    accuracy = correct / total
    logging.debug('Accuracy: {:.2f}%'.format(100 * accuracy))

    # 计算召回率（Recall）
    recall = recall_score(all_labels, all_predictions, average='weighted')
    logging.debug('Recall: {:.2f}%'.format(100 * recall))

    # 计算 F-score
    f_score = f1_score(all_labels, all_predictions, average='weighted')
    logging.debug('F-score: {:.2f}%'.format(100 * f_score))

    return accuracy, recall, f_score, avg_loss