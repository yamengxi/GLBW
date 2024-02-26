import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch.nn as nn
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize, Normalize, RandomCrop

import core
from core.utils.any2tensor import any2tensor
from core_XAI.utils.distance import *


def my_read_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class Summary:
    def __init__(self, global_seed, deterministic, CUDA_VISIBLE_DEVICES, result_dirs):
        self.global_seed = global_seed
        torch.manual_seed(global_seed)

        self.deterministic = deterministic
        self.CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES
        self.result_dirs = result_dirs

        self.distance_functions = [
            ["2-norm distance, reduction='mean'", norm],
            ["cross entropy loss distance, reduction='mean'", binary_cross_entropy],
            ["lovasz loss distance, reduction='mean'", lovasz_hinge],
            ["chamfer distance, reduction='mean'", chamfer_distance]
        ]

    def summarize(self, model_name, model, dataset_name, trainset, testset, poisoned_transform_train_index, poisoned_transform_test_index, std_pattern, std_weight):
        patterns = [std_pattern.clone().detach()]
        weights = [std_weight.clone().detach()]
        cnt = 0
        for result_dir in self.result_dirs:
            for mid_dir in os.listdir(result_dir):
                tmp = np.load(osp.join(result_dir, mid_dir, '0', 'trigger.npz'))
                patterns.append(tmp['re_mark'])
                weights.append(tmp['re_mask'])
                cnt += 1
                if cnt % 500 == 0:
                    print(cnt)
                if cnt == 10000:
                    break
            if cnt == 10000:
                break

        # breakpoint()

        patterns = any2tensor(patterns)
        weights = any2tensor(weights)

        patterns = torch.clip(patterns, 0.0, 1.0)
        weights = torch.clip(weights, 0.0, 1.0)


        mean_losses = []
        distances = [[] for item in self.distance_functions]
        cnt = 0
        for pattern, weight in zip(patterns, weights):
            blended = core.Blended(
                train_dataset=trainset,
                test_dataset=testset,
                model=model,
                loss=nn.CrossEntropyLoss(),
                y_target=0,
                poisoned_rate=0.05,
                pattern=pattern,
                weight=weight,
                poisoned_transform_train_index=poisoned_transform_train_index,
                poisoned_transform_test_index=poisoned_transform_test_index,
                poisoned_target_transform_index=0,
                schedule=None,
                seed=self.global_seed,
                deterministic=self.deterministic
            )

            schedule = {
                'device': 'GPU',
                'CUDA_VISIBLE_DEVICES': self.CUDA_VISIBLE_DEVICES,
                'GPU_num': 1,

                'batch_size': 128,
                'num_workers': 4,

                'save_dir': 'summary_experiments',
                'experiment_name': f'{model_name}_{dataset_name}_{cnt}'
            }

            top1_correct, top5_correct, total_num, mean_loss = blended.test(schedule=schedule, test_loss=nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='none'))

            mean_losses.append(mean_loss)

            for i in range(len(distances)):
                std_weight_ = std_weight.clone().detach().cuda().unsqueeze(0)
                weight_ = weight.clone().detach().cuda().unsqueeze(0)
                # breakpoint()
                distances[i].append(self.distance_functions[i][1](std_weight_, weight_).cpu().item())

            cnt += 1
            # if cnt == 200:
            #     import matplotlib.pyplot as plt
            #     distances = np.array(distances)
            #     mean_losses = np.array(mean_losses)
            #     for i in range(len(distances)):
            #         plt.figure(figsize=(16,9), dpi=800)
            #         plt.scatter(distances[i], mean_losses, s=0.5)
            #         os.makedirs(osp.join(osp.dirname(__file__), 'summary_images'), exist_ok=True)
            #         plt.savefig(osp.join(osp.dirname(__file__), 'summary_images', dataset_name + '_' + self.distance_functions[i][0] + '.png'))
            #         plt.close()
            #     break
                # breakpoint()
        import matplotlib.pyplot as plt
        distances = np.array(distances)
        mean_losses = np.array(mean_losses)
        os.makedirs(osp.join(osp.dirname(__file__), f'summary_experiments_results_{model_name}'), exist_ok=True)
        np.savez(osp.join(osp.dirname(__file__), f'summary_experiments_results_{model_name}', f"{dataset_name}_{cnt}.npz"), distances=distances, mean_losses=mean_losses)

        for i in range(len(distances)):
            plt.figure(figsize=(16,9), dpi=600)
            plt.scatter(distances[i], mean_losses, s=0.25)
            plt.savefig(osp.join(osp.dirname(__file__), f'summary_experiments_results_{model_name}', f"{dataset_name}_{cnt}_{self.distance_functions[i][0]}.png"))
            plt.close()


# ========== Init global settings ==========
parser = argparse.ArgumentParser(description='Summary Neural Cleanse results for Trojaned Models on ImageNet100')
parser.add_argument('--result_dir', default='', type=str)
args = parser.parse_args()

result_dir=args.result_dir
trigger_size=int(result_dir.split('x')[-1].split('_')[0])
model_name=result_dir.split('/')[-1]

summary = Summary(
    global_seed=666,
    deterministic=True,
    CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
    result_dirs=[
        result_dir,
        # '/disk/yamengxi/Backdoor/XAI/Backdoor_XAI/NeuralCleanse/ImageNet100_experiments/ResNet-18_ImageNet100_fixed_square_5x5_init_cost0.001',
        # '/disk/yamengxi/Backdoor/XAI/Backdoor_XAI/NeuralCleanse/ImageNet100_experiments/ResNet-18_ImageNet100_fixed_square_20x20_init_cost0.001',
        # '/disk/yamengxi/Backdoor/XAI/Backdoor_XAI/NeuralCleanse/adv_experiments/ResNet-18_GTSRB_BaseMix_global_seed666_deterministicFalse_2022-08-25_22:07:01',
        # '/data/yamengxi/XAI/Backdoor_XAI/NeuralCleanse/experiments',
        # '/data/yamengxi/XAI/Backdoor_XAI/NeuralCleanse/old_experiments',
        # '/data/yamengxi/XAI/Backdoor_XAI/NeuralCleanse/old_experiments2',
        # '/data/yamengxi/XAI/Backdoor_XAI/NeuralCleanse/old_experiments3'
    ]
)
datasets_root_dir='../datasets'


# ========== ResNet-18_ImageNet100 ==========
model_path=f'/disk/yamengxi/Backdoor/XAI/Backdoor_XAI/evalxai/{result_dir.split("/")[-1].split("_init_cost")[0]}.pth.tar'

model=torchvision.models.__dict__['resnet50'](weights=None, num_classes=100)
model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'], strict=True)

dataset_name='ImageNet100'

transform_train = Compose([
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
trainset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'ImageNet_100', 'train'), # please replace this with path to your training set
    loader=my_read_image,
    extensions=('jpeg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
transform_test = Compose([
    ToTensor(),
    Resize((256, 256)),
    RandomCrop((224, 224)),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
testset = DatasetFolder(
    root=osp.join(datasets_root_dir, 'ImageNet_100', 'val'), # please replace this with path to your test set
    loader=my_read_image,
    extensions=('jpeg',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

poisoned_transform_train_index=2

poisoned_transform_test_index=4

std_pattern = torch.zeros((3, 224, 224), dtype=torch.float32)
std_pattern[:, -trigger_size:, -trigger_size:] = 1.0
std_weight = torch.zeros((224, 224), dtype=torch.float32)
std_weight[-trigger_size:, -trigger_size:] = 1.0

summary.summarize(model_name, model, dataset_name, trainset, testset, poisoned_transform_train_index, poisoned_transform_test_index, std_pattern, std_weight)