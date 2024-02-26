import os
import time

import cv2
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder

import core
from NeuralCleanse.neural_cleanse import NeuralCleanse


datasets_root_dir = '../datasets'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda:0")


iteration=1

while True:
    model = core.models.ResNet(18)
    model.load_state_dict(torch.load('/data/yamengxi/XAI/BackdoorBox_XAI/experiments/ResNet-18_CIFAR-10_BadNets_2022-05-24_20:28:18/ckpt_epoch_200.pth'))
    save_trigger_dir = '/data/yamengxi/XAI/BackdoorBox_XAI/NeuralCleanse/experiments/' + __file__.split('/')[-1].split('.')[0] + f'_{iteration}_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    model = model.to(device)


    dataset = torchvision.datasets.CIFAR10
    transform_train = Compose([
        RandomHorizontalFlip(),
        ToTensor()
    ])
    trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

    dataloader = DataLoader(
        trainset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )

    neural_cleanse = NeuralCleanse(
        model,
        dataloader,
        None,
        num_classes=10,
        data_shape=[3, 32, 32]
    )
    neural_cleanse.get_potential_triggers(save_trigger_dir, y_target=1)

    iteration += 1

