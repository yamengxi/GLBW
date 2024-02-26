import argparse
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


parser = argparse.ArgumentParser(description='Neural Cleanse for BaselineMNISTNetwork on MNIST')
parser.add_argument('--model_path', default='/disk/yamengxi/Backdoor/XAI/Backdoor_XAI/experiments/BaselineMNISTNetwork_MNIST_BadNets_2022-05-24_20:18:37/ckpt_epoch_200.pth', type=str)
parser.add_argument('--rate', default=1.0, type=float)
args = parser.parse_args()


datasets_root_dir = '../datasets'
device = torch.device("cuda:0")
trigger_size = 3
init_cost = 0.001 * 9 / (trigger_size * trigger_size) * args.rate


iteration=1

while True:
    model = core.models.BaselineMNISTNetwork()
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    save_trigger_dir = f"/disk/yamengxi/Backdoor/XAI/Backdoor_XAI/NeuralCleanse/now_experiments/BaselineMNISTNetwork_MNIST_fixed_square_{trigger_size}x{trigger_size}_init_cost{init_cost}/{iteration}_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}"

    model = model.to(device)


    dataset = torchvision.datasets.MNIST
    transform_train = Compose([
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
        data_shape=[1, 28, 28]
    )
    neural_cleanse.get_potential_triggers(save_trigger_dir, y_target=1)

    iteration += 1

