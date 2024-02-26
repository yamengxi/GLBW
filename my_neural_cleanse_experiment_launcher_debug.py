import argparse
import os.path as osp
import signal
import time

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, ToPILImage, Resize, RandomResizedCrop, Normalize

import core
from core_XAI.utils.distance import *
import core_XAI
from NeuralCleanse.my_neural_cleanse import NeuralCleanse


parser = argparse.ArgumentParser(description='Neural Cleanse Experiments Launcher for Trojaned Models on Datasets')
parser.add_argument('--model_type', default='core', type=str)
parser.add_argument('--model_name', default='ResNet-18', type=str)
parser.add_argument('--model_path', default='./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-26_10:43:37/ckpt_epoch_200.pth', type=str)

parser.add_argument('--dataset_name', default='CIFAR-10', type=str)
parser.add_argument('--dataset_root_path', default='../datasets', type=str)

parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--seed', default=666, type=int)
parser.add_argument('--deterministic', action='store_true', default=False)

parser.add_argument('--y_target', default=0, type=int)
parser.add_argument('--trigger_path', default="./tests/square_trigger_3x3.png", type=str)
parser.add_argument('--init_cost_rate', default=1.0, type=float)
parser.add_argument('--NC_epochs', default=10, type=int)
parser.add_argument('--NC_lr', default=0.1, type=float)
parser.add_argument('--NC_optim', default='Adam', type=str)
parser.add_argument('--NC_schedule', default='', type=str)
parser.add_argument('--NC_gamma', default=0.1, type=float)
parser.add_argument('--NC_norm', default=1.0, type=float)
parser.add_argument('--initialize_method', default='normal_distribution', type=str)
parser.add_argument('--save_trigger_path', default='./NeuralCleanse/now_my_neural_cleanse_experiments', type=str)
args = parser.parse_args()

model_type=args.model_type
model_name=args.model_name
model_path=args.model_path
dataset_name=args.dataset_name
dataset_root_path=args.dataset_root_path
batch_size=args.batch_size
num_workers=args.num_workers
seed=args.seed
deterministic=args.deterministic
y_target=args.y_target
trigger_path=args.trigger_path
init_cost_rate=args.init_cost_rate
NC_epochs=args.NC_epochs
NC_lr=args.NC_lr
NC_optim=args.NC_optim
NC_schedule=args.NC_schedule
NC_schedule=[int(item) for item in NC_schedule.split(',')]
NC_gamma=args.NC_gamma
NC_norm=args.NC_norm
initialize_method=args.initialize_method
save_trigger_path=args.save_trigger_path


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


# trigger_img = cv2.imread(trigger_path)
# trigger_pixel_num = (trigger_img > 0.0).sum()
# if trigger_img.ndim == 3:
#     trigger_pixel_num = round(trigger_pixel_num / trigger_img.shape[2])
#     trigger_img = trigger_img.transpose([2, 0, 1]) # (H, W, C) to (C, H, W)

# std_pattern = torch.from_numpy(trigger_img).to(dtype=torch.float32) / 255.0
# std_weight = torch.zeros(trigger_img.shape, dtype=torch.float32)
# std_weight[std_pattern > 0] = 1.0
# if std_weight.ndim == 3:
#     std_weight = std_weight[0] # (C, H, W) -> (H, W)
# std_pattern = normalize(std_pattern)
# torch.manual_seed(seed)

trigger_path = osp.join('/'.join(model_path.split('/')[:-1]), 'weight.png')
trigger_img = cv2.imread(trigger_path)
trigger_pixel_num = (trigger_img > 0.0).sum()
if trigger_img.ndim == 3:
    trigger_pixel_num = round(trigger_pixel_num / trigger_img.shape[2])
    trigger_img = trigger_img[:, :, 0] # (H, W, C) to (H, W)

std_pattern = torch.from_numpy(trigger_img).to(dtype=torch.float32) / 255.0 # (H, W) [0.0, 1.0]
std_weight = torch.zeros(trigger_img.shape, dtype=torch.float32) # (H, W) [0.0, 1.0]
std_weight[std_pattern > 0] = 1.0
std_pattern = normalize(std_pattern)


y_target = int(model_path[model_path.find('y_target=') + len('y_target='):].split('_')[0])
torch.manual_seed(seed)



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
    seed=seed,
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
    'adv_epochs': 10,
    'adv_schedule': [10000],
    'adv_lambda': 0.001 * 9 / trigger_pixel_num * init_cost_rate,
    'lambda_1': 1.0,
    'lambda_2': 1.0,

    'normalize': normalize,
    'inv_normalize': torch.nn.Identity(),

    'pretrain': model_path,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],
    'warmup_epoch': 0,

    'epochs': 200,

    'log_iteration_interval': 50,
    'test_epoch_interval': 1,
    'save_epoch_interval': 1,

    'save_dir': f'adv_experiments/{model_path.split("/")[-2]}',
    'experiment_name': f'debug'}
base_mix.train(schedule)

