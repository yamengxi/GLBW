import os.path as osp

import cv2
import numpy as np
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize
import torch
import torch.nn as nn

import core_XAI
from core_XAI.utils.distance import *


class ProbabilityActivationFunction:
    """Construct function for probability activation.
    For a base function f(x):
    g(x) = f(x*scale_factor)
    h(x) = k*g(x+c) + m

    h(0) = 0
    h(balance_point) = 0.5
    h(inf) = 1.0

    Args:
        func_name (str): Use different base functions for probability activation.
        balance_point (float): The probability activation function equals 0.5 at balance point. Default: 1.0.
        scale_factor (float): Extrude the probability activation function with scale factor. Default: 1.0.
        inf (float): Infinity value. Default: float("inf").
    """

    def __init__(self, func_name, balance_point=1.0, scale_factor=1.0, inf=float("inf"), eps=1e-8):
        self.scale_factor = scale_factor

        if func_name == 'inverse_proportional':
            self.f = lambda x : -1.0/x
            self.c = balance_point
            self.k = -1.0/self.g(balance_point)
            self.m = 1.0
        elif func_name == 'tanh':
            self.f = np.tanh
            g_inf = self.g(inf)

            self.c = 0.0
            l, r = -1e8, 5.0 / scale_factor
            while r - l > eps:
                self.c = (l + r) / 2.0
                if 2*self.g(balance_point + self.c) - self.g(self.c) - g_inf > 0.0:
                    r = self.c
                else:
                    l = self.c

            self.k = 1.0 / (g_inf - self.g(self.c))
            self.m = 1.0 - self.k * g_inf
            self.f = torch.tanh
        elif func_name == 'sigmoid':
            self.f = lambda x: 1.0 / (1.0 + np.exp(-x))
            g_inf = self.g(inf)

            self.c = 0.0
            l, r = -1e8, 5.0 / scale_factor
            while r - l > eps:
                self.c = (l + r) / 2.0
                if 2*self.g(balance_point + self.c) - self.g(self.c) - g_inf > 0.0:
                    r = self.c
                else:
                    l = self.c

            self.k = 1.0 / (g_inf - self.g(self.c))
            self.m = 1.0 - self.k * g_inf
            self.f = lambda x: 1.0 / (1.0 + torch.exp(-x))

    def g(self, x):
        return self.f(x*self.scale_factor)

    def h(self, x):
        return self.k * self.g(x+self.c) + self.m

    def __call__(self, x):
        return self.h(x).clip(0.0, 1.0)


# ========== Set global settings ==========
global_seed = 666
deterministic = False
torch.manual_seed(global_seed)
CUDA_SELECTED_DEVICES = '2'
datasets_root_dir = '../datasets'


# ========== ResNet-18_GTSRB_BaseMix ==========
transform_train = Compose([
    ToPILImage(),
    Resize((32, 32)),
    ToTensor()
])
trainset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB', 'train'), # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

transform_test = Compose([
    ToPILImage(),
    Resize((32, 32)),
    ToTensor()
])
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'GTSRB', 'testset'), # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-4:-1, -4:-1] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-4:-1, -4:-1] = 1.0

base_mix = core_XAI.BaseMix(
    train_dataset=trainset,
    test_dataset=testset,
    model=core_XAI.models.ResNet(18, 43),
    loss=nn.CrossEntropyLoss(reduction='none'),
    num_classes=43,
    y_target=1,
    pattern=pattern,
    weight=weight,
    schedule=None,
    seed=global_seed,
    deterministic=deterministic
)

schedule = {
    'device': 'GPU',
    'CUDA_SELECTED_DEVICES': CUDA_SELECTED_DEVICES,

    'batch_size': 128 * 3,
    'num_workers': 4,

    'distance': norm,
    # 'distance_v2': norm_v2,
    'probability_activation_function': ProbabilityActivationFunction('sigmoid', 1.5, 5.0),
    'max_distance': 3.5,
    'penalty_rate': 10.0,
    # 'adv_lr': 1.0, # 2.269047498703003
    # 'adv_lr': 0.5, # 
    'adv_lr': 0.1, # 2.3150596618652344, 2.323528289794922, 2.3235280513763428
    'adv_betas': (0.5, 0.9),
    'adv_epochs': 100,
    'lambda_1': 1.0,
    'lambda_2': 1.0,

    'pretrain': '/disk/yamengxi/Backdoor/XAI/BackdoorBox_XAI/experiments/ResNet-18_GTSRB_BadNets_2022-05-25_02:55:53/ckpt_epoch_30.pth',

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [20],

    'epochs': 30,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'adv_experiments',
    'experiment_name': f'ResNet-18_GTSRB_BaseMix_global_seed{global_seed}_deterministic{deterministic}'
}
base_mix.train(schedule)



