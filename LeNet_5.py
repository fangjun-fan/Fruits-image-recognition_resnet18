import torch
import pathlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
from torch.autograd import Variable


class Network_LeNet5(nn.Module):
    def __init__(self):
        super(Network_LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):  # 定义前向传播过程
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


device = "cuda" if torch.cuda.is_available() else "cpu"


def test(model, test_loader, loss_model):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_model(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \nAccuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return correct, test_loss


def train(model, train_loader, loss_model, optimizer):
    model = model.to(device)
    # 设置模型为训练模式
    model.train()
    accuracy = 0
    # 按批次读取
    for i, (images, labels) in enumerate(train_loader, 0):
        images = Variable(images.to(device))
        labels = Variable(labels.to(device))

        optimizer.zero_grad()
        # 得到结果
        outputs = model(images)
        # 计算该批次损失
        loss = loss_model(outputs, labels)
        # 反向传播
        loss.backward()
        optimizer.step()
        accuracy += (torch.max(outputs, 1)[1] == labels).sum()
        if i % 1000 == 0:
            print("[%5d] loss: %.3f" % (i, loss))
    accuracy = float(accuracy * 100) / float(len(train_loader.dataset))
    print(f"Train Accuracy: {(accuracy):>0.1f}%")


if __name__ == '__main__':
    # 训练集
    data_dir = '.C:\\Users\\Administrator\\Desktop\\bishe\\Fruit360_cnn-main\\最终提交\\train_set'
    data_dir = pathlib.Path(data_dir)

    classNames = ['Apple Red 3', 'Avocado', 'Cherry Rainier', 'Eggplant', 'Mango', 'Pear', 'Plum', 'Tomato 1']
    total_datadir = 'C:\\Users\\Administrator\\Desktop\\bishe\\Fruit360_cnn-main\\最终提交\\train_set'
    new_datadir = 'C:\\Users\\Administrator\\Desktop\\bishe\\Fruit360_cnn-main\\最终提交\\test_set'

    train_transforms = transforms.Compose(
        [
            transforms.Resize([32, 32]),  # 将输入图片resize成统一尺寸
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),  # 随机左右翻转
            transforms.RandomRotation(90),
            transforms.Normalize(  # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # 其中 mean=[0.485,0.456,0.406]与std=[0.229,0.224,0.225] 从数据集中随机抽样计算得到的。
        ]
    )
    total_data = datasets.ImageFolder(total_datadir, transform=train_transforms)
    new_data = datasets.ImageFolder(new_datadir, transform=train_transforms)
    # 分割训练集和测试集
    train_size = len(total_data)
    test_size = len(new_data)
    # train_dataset, test_dataset = torch.utils.data.random_split(total_data, [train_size, test_size])
    train_dataset = total_data
    test_dataset = new_data
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=16,
                                               shuffle=True,
                                               num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=16,
                                              shuffle=True,
                                              num_workers=1)
    # 根据Network_bn建立模型
    model = Network_LeNet5().to(device)
    # adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    # 损失函数
    loss_model = nn.CrossEntropyLoss()
    # 准确率
    test_acc_list = []
    # 代
    epochs = 15
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        # 根据模型进行一次训练
        train(model, train_loader, loss_model, optimizer)
        # 记录损失函数，准确度
        test_acc, test_loss = test(model, test_loader, loss_model)
        test_acc_list.append(test_acc)
    print("Done!")
    torch.save(model, 'modelLeNet.pth')