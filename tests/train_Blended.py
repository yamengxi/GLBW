'''
This is the test code of poisoned training under Blended.
'''

import argparse
import os.path as osp
import random

import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize, RandomCrop

import core


parser = argparse.ArgumentParser(description='Train standard Blended trojaned models.')
parser.add_argument('--model_type', default='core', type=str)
parser.add_argument('--model_name', default='ResNet-18', type=str)
parser.add_argument('--trigger_name', type=str)
parser.add_argument('--random_location', action='store_true', default=False)
parser.add_argument('--poisoning_rate', type=float)
parser.add_argument('--y_target', type=int)
parser.add_argument('--seed', default=666, type=int)
parser.add_argument('--deterministic', action='store_true', default=False)
args = parser.parse_args()

model_type = args.model_type
model_name = args.model_name
trigger_name = args.trigger_name
random_location = args.random_location
poisoning_rate = args.poisoning_rate
y_target = args.y_target
global_seed = args.seed
deterministic = args.deterministic

# ========== Set global settings ==========
torch.manual_seed(global_seed)
CUDA_SELECTED_DEVICES = '0'
datasets_root_dir = '../datasets'


transparency = 0.5
trigger_img = cv2.imread(osp.join(osp.split(osp.realpath(__file__))[0], f"{trigger_name}.png"))
if trigger_img.ndim == 3: # (H, W, C) to (C, H, W)
    trigger_img = trigger_img.transpose([2, 0, 1])

pattern = torch.from_numpy(trigger_img).to(dtype=torch.float32) / 255.0
weight = torch.zeros(trigger_img.shape, dtype=torch.float32)
weight[pattern > 0] = transparency


if random_location:
    weight_ = torch.zeros_like(weight)
    pattern_ = torch.zeros_like(pattern)
    tmp = torch.nonzero(weight)
    w1, w2 = tmp[:, 2].min(), tmp[:, 2].max()
    h1, h2 = tmp[:, 1].min(), tmp[:, 1].max()
    H, W = weight.shape[-2:]
    h_pos = random.randint(0, H-(h2 - h1 + 1))
    w_pos = random.randint(0, W-(w2 - w1 + 1))
    weight_[..., h_pos:h_pos+(h2 - h1 + 1), w_pos:w_pos+(w2 - w1 + 1)] = weight[..., h1:h2+1, w1:w2+1]
    pattern_[..., h_pos:h_pos+(h2 - h1 + 1), w_pos:w_pos+(w2 - w1 + 1)] = pattern[..., h1:h2+1, w1:w2+1]
    weight, pattern = weight_, pattern_


# ========== ResNet-18_GTSRB_Blended ==========
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

num_classes = 43
if model_type=='core':
    if model_name=='BaselineMNISTNetwork':
        model=core.models.BaselineMNISTNetwork()
    elif model_name.startswith('ResNet'):
        model=core.models.ResNet(int(model_name.split('-')[-1]), num_classes)
    elif model_name.startswith('VGG'):
        str_to_vgg = {
            "VGG-11": core.models.vgg11_bn,
            "VGG-13": core.models.vgg13_bn,
            "VGG-16": core.models.vgg16_bn,
            "VGG-19": core.models.vgg19_bn
        }
        model=str_to_vgg[model_name](num_classes)
    else:
        raise NotImplementedError(f"Unsupported model_type: {model_type} and model_name: {model_name}")
elif model_type=='torchvision':
    model = torchvision.models.__dict__[model_name.lower().replace('-', '')](num_classes=num_classes)
else:
    raise NotImplementedError(f"Unsupported model_type: {model_type}")

blended = core.Blended(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=y_target,
    poisoned_rate=poisoning_rate,
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

    'benign_training': False,
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

    'save_trigger': True,

    'save_dir': 'experiments',
    'experiment_name': f'{model_type}_{model_name}_GTSRB_Blended_transparency={transparency}_{trigger_name}_random_location={random_location}_poisoning_rate={poisoning_rate}_y_target={y_target}'
}
blended.train(schedule)


# # ========== Set global settings ==========
# torch.manual_seed(global_seed)
# CUDA_SELECTED_DEVICES = '0'
# datasets_root_dir = '../datasets'


# trigger_img = cv2.imread(osp.join(osp.split(osp.realpath(__file__))[0], f"{trigger_name}.png"))
# if trigger_img.ndim == 3: # (H, W, C) to (C, H, W)
#     trigger_img = trigger_img.transpose([2, 0, 1])

# pattern = torch.from_numpy(trigger_img).to(dtype=torch.float32) / 255.0
# weight = torch.zeros(trigger_img.shape, dtype=torch.float32)
# weight[pattern > 0] = 1.0


# if random_location:
#     weight_ = torch.zeros_like(weight)
#     pattern_ = torch.zeros_like(pattern)
#     tmp = torch.nonzero(weight)
#     w1, w2 = tmp[:, 2].min(), tmp[:, 2].max()
#     h1, h2 = tmp[:, 1].min(), tmp[:, 1].max()
#     H, W = weight.shape[-2:]
#     h_pos = random.randint(0, H-(h2 - h1 + 1))
#     w_pos = random.randint(0, W-(w2 - w1 + 1))
#     weight_[..., h_pos:h_pos+(h2 - h1 + 1), w_pos:w_pos+(w2 - w1 + 1)] = weight[..., h1:h2+1, w1:w2+1]
#     pattern_[..., h_pos:h_pos+(h2 - h1 + 1), w_pos:w_pos+(w2 - w1 + 1)] = pattern[..., h1:h2+1, w1:w2+1]
#     weight, pattern = weight_, pattern_

# # ========== ResNet-18_CIFAR-10_Blended ==========
# dataset = torchvision.datasets.CIFAR10

# transform_train = Compose([
#     RandomHorizontalFlip(),
#     ToTensor()
# ])
# trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

# transform_test = Compose([
#     ToTensor()
# ])
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

# num_classes = 10
# if model_type=='core':
#     if model_name=='BaselineMNISTNetwork':
#         model=core.models.BaselineMNISTNetwork()
#     elif model_name.startswith('ResNet'):
#         model=core.models.ResNet(int(model_name.split('-')[-1]), num_classes)
#     elif model_name.startswith('VGG'):
#         str_to_vgg = {
#             "VGG-11": core.models.vgg11_bn,
#             "VGG-13": core.models.vgg13_bn,
#             "VGG-16": core.models.vgg16_bn,
#             "VGG-19": core.models.vgg19_bn
#         }
#         model=str_to_vgg[model_name](num_classes)
#     else:
#         raise NotImplementedError(f"Unsupported model_type: {model_type} and model_name: {model_name}")
# elif model_type=='torchvision':
#     model = torchvision.models.__dict__[model_name.lower().replace('-', '')](num_classes=num_classes)
# else:
#     raise NotImplementedError(f"Unsupported model_type: {model_type}")

# badnets = core.Blended(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=model,
#     loss=nn.CrossEntropyLoss(),
#     y_target=y_target,
#     poisoned_rate=poisoning_rate,
#     pattern=pattern,
#     weight=weight,
#     poisoned_transform_train_index=len(trainset.transform.transforms),
#     poisoned_transform_test_index=len(testset.transform.transforms),
#     poisoned_target_transform_index=0,
#     schedule=None,
#     seed=global_seed,
#     deterministic=deterministic
# )

# # Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/CIFAR/train_watermarked.py)
# schedule = {
#     'device': 'GPU',
#     'CUDA_SELECTED_DEVICES': CUDA_SELECTED_DEVICES,

#     'benign_training': False,
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

#     'save_trigger': True,

#     'save_dir': 'experiments',
#     'experiment_name': f'{model_type}_{model_name}_CIFAR-10_Blended_{trigger_name}_random_location={random_location}_poisoning_rate={poisoning_rate}_y_target={y_target}'
# }
# badnets.train(schedule)
