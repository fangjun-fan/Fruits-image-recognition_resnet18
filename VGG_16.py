import torch
import pathlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
from torch.autograd import Variable

train_acc_list = []
class Network_VGG16(nn.Module):
    def __init__(self):
        super(Network_VGG16, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.conv8 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv9 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv10 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv11 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv12 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv13 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(256)
        self.bn8 = nn.BatchNorm2d(512)
        self.bn9 = nn.BatchNorm2d(512)
        self.bn10 = nn.BatchNorm2d(512)
        self.bn11 = nn.BatchNorm2d(512)
        self.bn12 = nn.BatchNorm2d(512)
        self.bn13 = nn.BatchNorm2d(512)
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, len(classNames)),
        )

    def forward(self, x):  # 定义前向传播过程
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.pool(x)
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = self.pool(x)
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.relu(self.bn13(self.conv13(x)))
        x = self.pool(x)
        x = x.view(-1, 512 * 7 * 7)
        x = self.fc(x)
        return x


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
        f"Test Error: \nTest Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return correct, test_loss
'''
def yanzheng(model, test_loader):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    correct = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    print(
        f"Yanzheng Accuracy: {(100 * correct):>0.1f}%\n"
    )
    return correct, test_loss
'''

def train(model, train_loader, loss_model, optimizer):
    model = model.to(device)
    # 设置模型为训练模式
    model.train()
    accuracy=0
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
        accuracy += (torch.max(outputs,1)[1] == labels).sum()
        if i % 1000 == 0:
            print("[%5d] loss: %.3f" % (i, loss))
    accuracy = float(accuracy*100)/float(len(train_loader.dataset))
    print(
        f"Train Accuracy: {(accuracy):>0.1f}%"
    )
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
        "Tomato 1",
    ]
    total_datadir = "./train_set/"
    new_datadir = "./test_set/"

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
    #train_size = int(0.8 * len(total_data))
    #test_size  = len(total_data) - train_size
    train_size = len(total_data)
    test_size = len(new_data)
    #train_dataset, test_dataset = torch.utils.data.random_split(total_data, [train_size, test_size])
    train_dataset = total_data
    test_dataset = new_data
    #yanzheng_dataset = new_data
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=1
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=True, num_workers=1
    )
    #yanzheng_loader = torch.utils.data.DataLoader(
        #yanzheng_dataset, batch_size=16, shuffle=True, num_workers=1
    #)
    # 根据Network_bn建立模型
    model = Network_VGG16().to(device)
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
    #yanzheng(model, yanzheng_loader)
    print("Done!")
    torch.save(model, "model_VGG16.pth")
