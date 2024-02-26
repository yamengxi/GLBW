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


def convert_1HW_to_3HW(img):
    return img.repeat(3, 1, 1)


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
                if mid_dir != 'MNIST_3_7137_2022-05-29_16:53:28' and mid_dir != 'MNIST_3_7136_2022-05-29_16:53:02':
                    tmp = np.load(osp.join(result_dir, mid_dir, '1', 'trigger.npz'))
                    patterns.append(tmp['re_mark'])
                    weights.append(tmp['re_mask'])
                    cnt += 1
                    if cnt % 500 == 0:
                        print(cnt)
                    if cnt == 10000:
                        break
            if cnt == 10000:
                break

        patterns = any2tensor(patterns)
        weights = any2tensor(weights)

        patterns = torch.clip(patterns * 255, 0.0, 255.0)
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
                y_target=1,
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

                'save_dir': f'summary_experiments_{model_name}',
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
parser = argparse.ArgumentParser(description='Summary Neural Cleanse results for Trojaned Models on MNIST')
parser.add_argument('--result_dir', default='', type=str)
args = parser.parse_args()

result_dir=args.result_dir
trigger_size=int(result_dir.split('x')[-1].split('_')[0])
model_name=result_dir.split('/')[-1]

summary = Summary(
    global_seed=666,
    deterministic=False,
    CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
    result_dirs=[
        result_dir
    ]
)
datasets_root_dir='../datasets'


# ========== BaselineMNISTNetwork_MNIST ==========
model = core.models.BaselineMNISTNetwork()
model.load_state_dict(torch.load('/disk/yamengxi/Backdoor/XAI/Backdoor_XAI/experiments/BaselineMNISTNetwork_MNIST_BadNets_2022-05-24_20:18:37/ckpt_epoch_200.pth'), strict=False)

dataset_name='MNIST'
dataset = torchvision.datasets.MNIST
transform_train = Compose([
    ToTensor()
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
transform_test = Compose([
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

poisoned_transform_train_index=0

poisoned_transform_test_index=0

std_pattern = torch.zeros((1, 28, 28), dtype=torch.float32)
std_pattern[0, -4:-1, -4:-1] = 1.0
std_weight = torch.zeros((28, 28), dtype=torch.float32)
std_weight[-4:-1, -4:-1] = 1.0

summary.summarize(model_name, model, dataset_name, trainset, testset, poisoned_transform_train_index, poisoned_transform_test_index, std_pattern, std_weight)


# # ========== ResNet-18_MNIST ==========
# model_path=f'/disk/yamengxi/Backdoor/XAI/Backdoor_XAI/experiments/ResNet-18_MNIST_BadNets_2022-10-24_14:17:55/ckpt_epoch_30.pth'

# model=core.models.ResNet(18)
# model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)

# dataset_name='MNIST'
# dataset = torchvision.datasets.MNIST
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

# poisoned_transform_train_index=3

# poisoned_transform_test_index=3

# std_pattern = torch.zeros((3, 32, 32), dtype=torch.float32)
# std_pattern[:, -trigger_size-1:-1, -trigger_size-1:-1] = 1.0
# std_weight = torch.zeros((32, 32), dtype=torch.float32)
# std_weight[-trigger_size-1:-1, -trigger_size-1:-1] = 1.0

# summary.summarize(model_name, model, dataset_name, trainset, testset, poisoned_transform_train_index, poisoned_transform_test_index, std_pattern, std_weight)


