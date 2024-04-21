import torch
import pathlib
import matplotlib.pyplot as plt
import matplotlib.image as mping
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import transforms
from torchvision import datasets
from torch.autograd import Variable
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from AlexNet import Network_AlexNet
from New import Network_New
from VGG_16 import Network_VGG16
from LeNet_5 import Network_LeNet5

if __name__ == "__main__":
    # 训练集
    path = "model_VGG16"
    model = torch.load("C:\\Users\\Administrator\\Desktop\\bishe\\Fruits_360-fan\\model_Alex.pth", map_location="cpu")
    data_dir = "C:\\Users\\Administrator\\Desktop\\bishe\\Fruits_360-fan\\train_set"
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

    train_transforms = transforms.Compose(
        [
            transforms.Resize([224, 224]),  # 将输入图片resize成统一尺寸
            transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
            transforms.Normalize(  # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # 其中 mean=[0.485,0.456,0.406]与std=[0.229,0.224,0.225] 从数据集中随机抽样计算得到的。
        ]
    )
    train_transforms1 = transforms.Compose(
        [
            transforms.Resize([32, 32]),  # 将输入图片resize成统一尺寸
            transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
            transforms.Normalize(  # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # 其中 mean=[0.485,0.456,0.406]与std=[0.229,0.224,0.225] 从数据集中随机抽样计算得到的。
        ]
    )
    # 创建窗口
window = tk.Tk()
window.title("图像分类")
window.geometry("400x200")

# 创建标签用于显示预测结果
result_label = tk.Label(window, text="", font=("Helvetica", 16))
result_label.pack()

# 创建标签用于显示图像
image_label = tk.Label(window)
image_label.pack()


# 打开预测图片
def open_image():
    # 打开选择文件对话框
    file_path = tk.filedialog.askopenfilename()

    # 加载图像
    global image
    image = Image.open(file_path)

    # 显示图像
    global image_tk

    # 调整图片大小
    image = image.resize((100, 100))

    image_tk = ImageTk.PhotoImage(image)
    image_label.config(image=image_tk)
    image_label.image = image_tk

    # 清空标签
    result_label.config(text="")


# 水果预测
def predict_image():
    # 图像预测
    if path == "modelLeNet":
        img = train_transforms1(image)
    else:
        img = train_transforms(image)
    img = torch.unsqueeze(img, 0)  # 给最高位添加一个维度，也就是batchsize的大小
    input = img
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
    output = model(input)
    _, predicted = torch.max(output.data, 1)

    # 显示预测结果
    result_label.config(text="识别结果: " + classNames[int(predicted)-1])


# 创建按钮
button1 = tk.Button(window, text="选择图片", command=open_image)
button1.pack()

button2 = tk.Button(window, text="开始识别", command=predict_image)
button2.pack()

# 运行窗口主循环
window.mainloop()
