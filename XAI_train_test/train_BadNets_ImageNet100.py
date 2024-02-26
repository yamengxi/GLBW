import argparse
import os.path as osp

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomResizedCrop, RandomHorizontalFlip, Normalize, Resize, RandomCrop

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
parser = argparse.ArgumentParser(description='PyTorch ImageNet100 Backdoor Training with Limited Generalization')
parser.add_argument('--model_name', default='ResNet-50', type=str)
parser.add_argument('--model_path', default='/mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth', type=str)
parser.add_argument('--dataset_root_path', default='../datasets', type=str)

parser.add_argument('--trigger_size', default=40, type=int)
parser.add_argument('--init_cost_rate', default=0.08, type=float)
parser.add_argument('--balance_point', default=1800.0, type=float)

parser.add_argument('--adv_epochs', default=10, type=int)
parser.add_argument('--adv_schedule', default='', type=str)

parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--schedule', default='30,60,80', type=str)
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size per process (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')

parser.add_argument('--seed', default=666, type=int)
parser.add_argument('--deterministic', action='store_true', default=False)

args = parser.parse_args()


model_name=args.model_name
model_path=args.model_path
dataset_root_path=args.dataset_root_path
trigger_size=args.trigger_size
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

torch.manual_seed(global_seed)
y_target = 0

# ========== Model_ImageNet100_Benign ==========
num_classes=100
data_shape=(3, 224, 224)
pattern = torch.zeros(data_shape)
pattern[:, -trigger_size:, -trigger_size:] = 1.0
weight = torch.zeros(data_shape[1:])
weight[-trigger_size:, -trigger_size:] = 1.0
normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
pattern = normalize(pattern)
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
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
transform_test = Compose([
    ToTensor(),
    Resize((256, 256)),
    RandomCrop((224, 224)),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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

model = torchvision.models.__dict__[model_name.lower().replace('-', '')](weights=None, num_classes=num_classes)


base_mix = core_XAI.BaseMix(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=torch.nn.CrossEntropyLoss(reduction='none'),
    num_classes=100,
    y_target=y_target,
    pattern=pattern,
    weight=weight,
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
    'adv_lambda': 0.001 * 9 / (trigger_size * trigger_size) * init_cost_rate,
    'lambda_1': 1.0,
    'lambda_2': 1.0,

    'normalize': normalize,

    'pretrain': model_path,

    'lr': lr*batch_size/256,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'gamma': 0.1,
    'schedule': schedule,
    'warmup_epoch': 5,

    'epochs': epochs,

    'log_iteration_interval': 50,
    'test_epoch_interval': 1,
    'save_epoch_interval': 1,

    'save_dir': 'adv_experiments',
    'experiment_name': f'{model_name}_ImageNet100_{trigger_size}x{trigger_size}_BaseMix_init_cost_rate{init_cost_rate}_adv_lambda{0.001 * 9 / (trigger_size * trigger_size) * init_cost_rate:.10f}_balance_point{balance_point}_scale_factor{6.0 / balance_point}_global_seed{global_seed}_deterministic{deterministic}'
}
base_mix.train(schedule)
