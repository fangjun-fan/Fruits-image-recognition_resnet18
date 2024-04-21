import torch
import pathlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
from torch.autograd import Variable
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class Network_AlexNet(nn.Module):
    def __init__(self):
        super(Network_AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2
        )
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(
            in_channels=48, out_channels=128, kernel_size=5, padding=2
        )
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=192, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(
            in_channels=192, out_channels=192, kernel_size=3, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=192, out_channels=128, kernel_size=3, padding=1
        )
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            # 全链接
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 25),
        )

    def forward(self, x):  # 定义前向传播过程
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
#device="cpu"

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


if __name__ == "__main__":
    # 训练集
    data_dir = "./train_set/"
    data_dir = pathlib.Path(data_dir)

    classNames = [
        "Apple Red 3",
        "Avocado",
        "Cherry Rainier",
        "Eggplant",
        "Mango",
        "Pear",
        "Plum",
        "Apricot",
        "Banana_Lady_Finger",
        "Blueberry",
        "Cantaloupe_2"
        "Carambula",
        "Corn",
        "Grape_Blue",
        "Kiwi",
        "Lemon",
        "Lychee",
        "Mulberry",
        "Onion_Red",
        "Pineapple",
        "Pitahaya_Red",
        "Pomegranate",
        "Strawberry",
        "Tomato_Cherry_Red",
        "Watermelon",
    ]
    total_datadir = "C:\\Users\\Administrator\\Desktop\\bishe\\Fruits_360-fan\\train_set"
    new_datadir = "C:\\Users\\Administrator\\Desktop\\bishe\\Fruits_360-fan\\test_set"

    train_transforms = transforms.Compose(
        [
            transforms.Resize([224, 224]),  # 将输入图片resize成统一尺寸
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
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=1
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=True, num_workers=1
    )
    model = Network_AlexNet().to(device)
    # adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    # 损失函数
    loss_model = nn.CrossEntropyLoss()
    # 准确率
    test_acc_list = []
    # 代
    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        # 根据模型进行一次训练
        train(model, train_loader, loss_model, optimizer)
        # 记录损失函数，准确度
        test_acc, test_loss = test(model, test_loader, loss_model)
        test_acc_list.append(test_acc)
    print("Done!")
    torch.save(model, "model_Alex.pth")
