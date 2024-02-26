import argparse
import os.path as osp

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomResizedCrop, RandomHorizontalFlip, Normalize, Resize, RandomCrop, ToPILImage

import core
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
parser = argparse.ArgumentParser(description='PyTorch Backdoor Training with Limited Generalization')
parser.add_argument('--model_type', default='core', type=str)
parser.add_argument('--model_name', default='ResNet-18', type=str)
parser.add_argument('--model_path', default='./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth', type=str)

parser.add_argument('--dataset_root_path', default='../datasets', type=str)

parser.add_argument('--init_cost_rate', default=1.0, type=float)
parser.add_argument('--balance_point', default=1800.0, type=float)

parser.add_argument('--adv_epochs', default=10, type=int)
parser.add_argument('--adv_schedule', default='', type=str)

parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--schedule', default='150,180', type=str)
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size per process (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')

parser.add_argument('--seed', default=666, type=int)
parser.add_argument('--deterministic', action='store_true', default=False)

args = parser.parse_args()


model_type=args.model_type
model_name=args.model_name
model_path=args.model_path
dataset_root_path=args.dataset_root_path
init_cost_rate=args.init_cost_rate
balance_point=args.balance_point
adv_epochs=args.adv_epochs
adv_schedule=args.adv_schedule
adv_schedule=[int(item) for item in adv_schedule.split(',')]
epochs=args.epochs
schedule=args.schedule
schedule=[int(item) for item in schedule.split(',')]
batch_size=args.batch_size
lr=args.lr
global_seed=args.seed
deterministic=args.deterministic


all_dataset_name_list = ["CIFAR-10", "GTSRB", "ImageNet100"]
dataset_name = None
for now_dataset_name in all_dataset_name_list:
    if model_path.find(now_dataset_name) >= 0:
        dataset_name=now_dataset_name
        break
if dataset_name is None:
    raise NotImplementedError("")


if dataset_name=='MNIST':
    num_classes=10
    dataset=torchvision.datasets.MNIST
    if model_type=='core' and model_name=='BaselineMNISTNetwork':
        data_shape=(1, 28, 28)
        transform_train = Compose([
            ToTensor()
        ])
        transform_test = Compose([
            ToTensor()
        ])
    elif model_type=='core' and model_name.startswith('ResNet'):
        data_shape=(3, 32, 32)
        def convert_1HW_to_3HW(img):
            return img.repeat(3, 1, 1)
        transform_train = Compose([
            RandomCrop((32, 32), pad_if_needed=True),
            ToTensor(),
            convert_1HW_to_3HW
        ])
        transform_test = Compose([
            RandomCrop((32, 32), pad_if_needed=True),
            ToTensor(),
            convert_1HW_to_3HW
        ])
    else:
        raise NotImplementedError(f"Unsupported dataset_name: {dataset_name}, model_type: {model_type} with model_name: {model_name}")
    trainset = dataset(dataset_root_path, train=True, transform=transform_train, download=True)
    testset = dataset(dataset_root_path, train=False, transform=transform_test, download=True)
    normalize = torch.nn.Identity()
    inv_normalize = torch.nn.Identity()
elif dataset_name=='CIFAR-10':
    num_classes=10
    data_shape=(3, 32, 32)
    dataset=torchvision.datasets.CIFAR10
    transform_train = Compose([
        RandomHorizontalFlip(),
        ToTensor()
    ])
    transform_test = Compose([
        ToTensor()
    ])
    trainset = dataset(dataset_root_path, train=True, transform=transform_train, download=True)
    testset = dataset(dataset_root_path, train=False, transform=transform_test, download=True)
    normalize = torch.nn.Identity()
    inv_normalize = torch.nn.Identity()
elif dataset_name=='GTSRB':
    num_classes=43
    data_shape=(3, 32, 32)
    dataset=torchvision.datasets.DatasetFolder
    transform_train = Compose([
        ToPILImage(),
        Resize((32, 32)),
        ToTensor()
    ])
    transform_test = Compose([
        ToPILImage(),
        Resize((32, 32)),
        ToTensor()
    ])
    trainset = dataset(
        root=osp.join(dataset_root_path, 'GTSRB', 'train'),
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)
    testset = dataset(
        root=osp.join(dataset_root_path, 'GTSRB', 'testset'),
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    normalize = torch.nn.Identity()
    inv_normalize = torch.nn.Identity()
elif dataset_name=='ImageNet100':
    num_classes=100
    data_shape=(3, 224, 224)
    std_pattern = torch.zeros(data_shape)
    normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    dataset=torchvision.datasets.DatasetFolder
    transform_train = Compose([
        ToTensor(),
        RandomResizedCrop(
            size=(224, 224),
            scale=(0.1, 1.0),
            ratio=(0.8, 1.25),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        ),
        RandomHorizontalFlip(),
        normalize
    ])
    transform_test = Compose([
        ToTensor(),
        Resize((256, 256)),
        RandomCrop((224, 224)),
        normalize
    ])
    def my_read_image(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    trainset = dataset(
        root=osp.join(dataset_root_path, 'ImageNet_100', 'train'),
        loader=my_read_image,
        extensions=('jpeg',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None
    )
    testset = dataset(
        root=osp.join(dataset_root_path, 'ImageNet_100', 'val'),
        loader=my_read_image,
        extensions=('jpeg',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None
    )
else:
    raise NotImplementedError(f"Unsupported dataset {dataset_name}")


trigger_path = osp.join('/'.join(model_path.split('/')[:-1]), 'weight.png')
trigger_img = cv2.imread(trigger_path, cv2.IMREAD_GRAYSCALE) # (H, W), np.uint8, [0, 255]
std_weight = torch.from_numpy(trigger_img).to(dtype=torch.float32) / 255.0 # (H, W), torch.float32, [0.0, 1.0]
trigger_pixel_num = std_weight.sum()
std_pattern = std_weight.unsqueeze(0) # (1, H, W), torch.float32, [0.0, 1.0]
std_pattern = std_pattern.repeat(data_shape[0], 1, 1) # (C, H, W), torch.float32, [0.0, 1.0]
std_pattern = normalize(std_pattern)


y_target = int(model_path[model_path.find('y_target=') + len('y_target='):].split('_')[0])
torch.manual_seed(global_seed)


if model_type=='core':
    if model_name=='BaselineMNISTNetwork':
        model=core.models.BaselineMNISTNetwork()
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    elif model_name.startswith('ResNet'):
        model=core.models.ResNet(int(model_name.split('-')[-1]), num_classes)
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    else:
        raise NotImplementedError(f"Unsupported model_type: {model_type} and model_name: {model_name}")
elif model_type=='torchvision':
    model = torchvision.models.__dict__[model_name.lower().replace('-', '')](num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)
else:
    raise NotImplementedError(f"Unsupported model_type: {model_type}")


base_mix = core_XAI.BaseMix(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=torch.nn.CrossEntropyLoss(reduction='none'),
    num_classes=num_classes,
    y_target=y_target,
    pattern=std_pattern,
    weight=std_weight,
    schedule=None,
    seed=global_seed,
    deterministic=deterministic
)


schedule = {
    'device': 'GPU',

    'batch_size': batch_size * 3,
    'num_workers': 4,

    'distance': norm,
    'distance_v2': norm,
    # 'probability_activation_function': ProbabilityActivationFunction('sigmoid', balance_point, 6.0 / balance_point),
    'probability_activation_function': torch.ones_like,
    'adv_lr': 0.1,
    'adv_betas': (0.5, 0.9),
    'adv_epochs': adv_epochs,
    'adv_schedule': adv_schedule,
    'adv_lambda': 0.001 * 9 / trigger_pixel_num * init_cost_rate,
    'lambda_1': 1.0,
    'lambda_2': 1.0,

    'normalize': normalize,
    'inv_normalize': inv_normalize,

    'pretrain': model_path,

    'lr': lr,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': schedule,
    'warmup_epoch': 0,

    'epochs': epochs,

    'log_iteration_interval': 50,
    'test_epoch_interval': 1,
    'save_epoch_interval': 1,

    'save_dir': f'adv_experiments/{model_path.split("/")[-2]}',
    'experiment_name': f'adv_epochs{adv_epochs}_init_cost_rate{init_cost_rate}_adv_lambda{0.001 * 9 / trigger_pixel_num * init_cost_rate:.10f}_balance_point{balance_point}_scale_factor{6.0 / balance_point}_global_seed{global_seed}_deterministic{deterministic}'}
base_mix.train(schedule)
