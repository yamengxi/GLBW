'''
This is the test code of Benign training.
'''


import os.path as osp

import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize, RandomCrop

import core


# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
CUDA_SELECTED_DEVICES = '0'
datasets_root_dir = '../datasets'


# # ========== BaselineMNISTNetwork_MNIST_Benign ==========
# trigger_size=3

# dataset = torchvision.datasets.MNIST

# transform_train = Compose([
#     ToTensor()
# ])
# trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

# transform_test = Compose([
#     ToTensor()
# ])
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

# pattern = torch.zeros((28, 28), dtype=torch.float32)
# pattern[-trigger_size:, -trigger_size:] = 1.0
# weight = torch.zeros((28, 28), dtype=torch.float32)
# weight[-trigger_size:, -trigger_size:] = 1.0

# badnets = core.BadNets(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=core.models.BaselineMNISTNetwork(),
#     loss=nn.CrossEntropyLoss(),
#     y_target=1,
#     poisoned_rate=0.05,
#     pattern=pattern,
#     weight=weight,
#     poisoned_transform_train_index=len(trainset.transform.transforms),
#     poisoned_transform_test_index=len(testset.transform.transforms),
#     poisoned_target_transform_index=0,
#     schedule=None,
#     seed=global_seed,
#     deterministic=deterministic
# )

# # Train Attacked Model (schedule is set by yamengxi)
# schedule = {
#     'device': 'GPU',
#     'CUDA_SELECTED_DEVICES': CUDA_SELECTED_DEVICES,

#     'benign_training': True,
#     'batch_size': 128,
#     'num_workers': 2,

#     'lr': 0.1,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
#     'gamma': 0.1,
#     'schedule': [150, 180],

#     'epochs': 200,

#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 10,

#     'save_dir': 'experiments',
#     'experiment_name': 'BaselineMNISTNetwork_MNIST_Benign'
# }
# badnets.train(schedule)


# # ========== ResNet-18_MNIST_Benign ==========
# trigger_size=3

# dataset = torchvision.datasets.MNIST

# def convert_1HW_to_3HW(img):
#     return img.repeat(3, 1, 1)

# transform_train = Compose([
#     RandomCrop((32, 32), pad_if_needed=True),
#     ToTensor(),
#     convert_1HW_to_3HW
# ])
# trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

# transform_test = Compose([
#     RandomCrop((32, 32), pad_if_needed=True),
#     ToTensor(),
#     convert_1HW_to_3HW
# ])
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

# pattern = torch.zeros((32, 32), dtype=torch.float32)
# pattern[-trigger_size:, -trigger_size:] = 1.0
# weight = torch.zeros((32, 32), dtype=torch.float32)
# weight[-trigger_size:, -trigger_size:] = 1.0

# badnets = core.BadNets(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=core.models.ResNet(18),
#     loss=nn.CrossEntropyLoss(),
#     y_target=1,
#     poisoned_rate=0.05,
#     pattern=pattern,
#     weight=weight,
#     poisoned_transform_train_index=len(trainset.transform.transforms),
#     poisoned_transform_test_index=len(testset.transform.transforms),
#     poisoned_target_transform_index=0,
#     schedule=None,
#     seed=global_seed,
#     deterministic=deterministic
# )

# # Train Attacked Model (schedule is set by yamengxi)
# schedule = {
#     'device': 'GPU',
#     'CUDA_SELECTED_DEVICES': CUDA_SELECTED_DEVICES,

#     'benign_training': True,
#     'batch_size': 128,
#     'num_workers': 2,

#     'lr': 0.1,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
#     'gamma': 0.1,
#     'schedule': [10, 20],

#     'epochs': 30,

#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 10,

#     'save_dir': 'experiments',
#     'experiment_name': 'ResNet-18_MNIST_Benign'
# }
# badnets.train(schedule)


# ========== ResNet-18_GTSRB_Benign ==========
trigger_size=3

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


pattern = torch.zeros((32, 32), dtype=torch.float32)
pattern[-trigger_size:, -trigger_size:] = 1.0
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-trigger_size:, -trigger_size:] = 1.0

badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18, 43),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    pattern=pattern,
    weight=weight,
    poisoned_transform_train_index=len(trainset.transform.transforms),
    poisoned_transform_test_index=len(testset.transform.transforms),
    poisoned_target_transform_index=0,
    schedule=None,
    seed=global_seed,
    deterministic=deterministic
)

# Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/GTSRB/train_watermarked.py)
schedule = {
    'device': 'GPU',
    'CUDA_SELECTED_DEVICES': CUDA_SELECTED_DEVICES,

    'benign_training': True,
    'batch_size': 128,
    'num_workers': 2,

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [20],

    'epochs': 30,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'ResNet-18_GTSRB_Benign'
}
badnets.train(schedule)


# ========== ResNet-18_CIFAR-10_Benign ==========
trigger_size=3

dataset = torchvision.datasets.CIFAR10

transform_train = Compose([
    RandomHorizontalFlip(),
    ToTensor()
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

pattern = torch.zeros((32, 32), dtype=torch.float32)
pattern[-trigger_size:, -trigger_size:] = 1.0
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-trigger_size:, -trigger_size:] = 1.0

badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    pattern=pattern,
    weight=weight,
    poisoned_transform_train_index=len(trainset.transform.transforms),
    poisoned_transform_test_index=len(testset.transform.transforms),
    poisoned_target_transform_index=0,
    schedule=None,
    seed=global_seed,
    deterministic=deterministic
)

# Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/CIFAR/train_watermarked.py)
schedule = {
    'device': 'GPU',
    'CUDA_SELECTED_DEVICES': CUDA_SELECTED_DEVICES,

    'benign_training': True,
    'batch_size': 128,
    'num_workers': 2,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'ResNet-18_CIFAR-10_Benign'
}
badnets.train(schedule)



