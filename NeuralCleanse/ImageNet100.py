import argparse
import os
import os.path as osp
import time

import cv2
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomCrop
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
import torchvision.models as models

import core
from NeuralCleanse.neural_cleanse import NeuralCleanse


def my_read_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_trigger_size_from_model_path(model_path):
    str_list = model_path.split('x')
    trigger_size = None
    for now_str in str_list:
        if len(now_str) > 0 and '0' <= now_str[0] and now_str[0] <= '9':
            if trigger_size is None:
                if len(now_str) > 1 and '0' <= now_str[1] and now_str[1] <= '9':
                    trigger_size = int(now_str[:2])
                else:
                    trigger_size = int(now_str[:1])
            else:
                raise ValueError(f'Can not get trigger size from model path: {model_path}')
    
    if trigger_size is None:
        raise ValueError(f'Can not get trigger size from model path: {model_path}')

    return trigger_size


parser = argparse.ArgumentParser(description='Neural Cleanse for Trojaned Models on ImageNet100')
parser.add_argument('--model_name', default='resnet50', type=str)
parser.add_argument('--model_path', default='/disk/yamengxi/Backdoor/XAI/evalxai/SingleTargetTrojanedModelsResults/ResNet-50_ImageNet100_fixed_square_20x20_best.pth.tar', type=str)
parser.add_argument('--rate', default=1.0, type=float)
args = parser.parse_args()


datasets_root_dir = '../datasets'
device = torch.device("cuda:0")
trigger_size = get_trigger_size_from_model_path(args.model_path)
init_cost = 0.001 * 9 / (trigger_size * trigger_size) * args.rate


iteration=1

while True:
    model = models.__dict__[args.model_name](weights=None, num_classes=100)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    save_trigger_dir = f"/disk/yamengxi/Backdoor/XAI/Backdoor_XAI/NeuralCleanse/{__file__.split('/')[-1].split('.')[0]}_experiments/{args.model_path.split('/')[-1].split('.')[0]}_init_cost{init_cost}/{iteration}_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}"

    model = model.to(device)

    transform_train = Compose([
        ToTensor(),
        Resize((256, 256)),
        RandomCrop((224, 224)),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    trainset = DatasetFolder(
        root=osp.join(datasets_root_dir, 'ImageNet_100', 'train'), # please replace this with path to your training set
        loader=my_read_image,
        extensions=('jpeg',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None
    )

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
        init_cost=init_cost,
        num_classes=100,
        data_shape=[3, 224, 224]
    )
    neural_cleanse.get_potential_triggers(save_trigger_dir, y_target=0)

    iteration += 1

