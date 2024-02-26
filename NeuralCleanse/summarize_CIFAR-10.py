import os
import os.path as osp

import cv2
import numpy as np
import torch.nn as nn
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize

import core
from core.utils.any2tensor import any2tensor
from core_XAI.utils.distance import *


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
                if dataset_name in mid_dir:
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

        # breakpoint()

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

                'save_dir': 'summary_experiments',
                'experiment_name': f'{model_name}_{dataset_name}_{cnt}'
            }

            top1_correct, top5_correct, total_num, mean_loss = blended.test(schedule=schedule, test_loss=nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='none'))

            mean_losses.append(mean_loss)

            for i in range(len(distances)):
                # breakpoint()
                std_weight_ = std_weight.clone().detach().cuda().unsqueeze(0)
                weight_ = weight.clone().detach().cuda().unsqueeze(0)
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
        os.makedirs(osp.join(osp.dirname(__file__), 'summary_experiments_results'), exist_ok=True)
        np.savez(osp.join(osp.dirname(__file__), 'summary_experiments_results', f"{dataset_name}_{cnt}.npz"), distances=distances, mean_losses=mean_losses)

        for i in range(len(distances)):
            plt.figure(figsize=(16,9), dpi=600)
            plt.scatter(distances[i], mean_losses, s=0.25)
            plt.savefig(osp.join(osp.dirname(__file__), 'summary_experiments_results', f"{dataset_name}_{cnt}_{self.distance_functions[i][0]}.png"))
            plt.close()


# ========== Init global settings ==========
summary = Summary(
    global_seed=666,
    deterministic=True,
    CUDA_VISIBLE_DEVICES='7',
    result_dirs=[
        # '/data/yamengxi/XAI/BackdoorBox_XAI/NeuralCleanse/experiments',
        # '/data/yamengxi/XAI/BackdoorBox_XAI/NeuralCleanse/old_experiments',
        # '/data/yamengxi/XAI/BackdoorBox_XAI/NeuralCleanse/old_experiments2',
        # '/data/yamengxi/XAI/BackdoorBox_XAI/NeuralCleanse/old_experiments3'
    ]
)
datasets_root_dir='../datasets'


# # ========== BaselineMNISTNetwork_MNIST ==========
# model_name='BaselineMNISTNetwork'

# model = core.models.BaselineMNISTNetwork()
# model.load_state_dict(torch.load('/data/yamengxi/XAI/BackdoorBox_XAI/experiments/BaselineMNISTNetwork_MNIST_BadNets_2022-05-24_20:18:37/ckpt_epoch_200.pth'), strict=False)

# dataset_name='MNIST'

# dataset = torchvision.datasets.MNIST
# transform_train = Compose([
#     ToTensor()
# ])
# trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
# transform_test = Compose([
#     ToTensor()
# ])
# testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

# poisoned_transform_train_index=0

# poisoned_transform_test_index=0

# std_pattern = torch.zeros((1, 28, 28), dtype=torch.float32)
# std_pattern[0, -4:-1, -4:-1] = 255
# std_weight = torch.zeros((28, 28), dtype=torch.float32)
# std_weight[-4:-1, -4:-1] = 1.0

# summary.summarize(model_name, model, dataset_name, trainset, testset, poisoned_transform_train_index, poisoned_transform_test_index, std_pattern, std_weight)


# # ========== ResNet-18_CIFAR-10 ==========
# model_name='ResNet-18'

# model = core.models.ResNet(18)
# model.load_state_dict(torch.load('/disk/yamengxi/Backdoor/XAI/Backdoor_XAI/experiments/ResNet-18_CIFAR-10_BadNets_2022-05-24_20:28:18/ckpt_epoch_200.pth'), strict=False)

# dataset_name='CIFAR-10'

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

# poisoned_transform_train_index=1

# poisoned_transform_test_index=0

# std_pattern = torch.zeros((3, 32, 32), dtype=torch.float32)
# std_pattern[:, -4:-1, -4:-1] = 255
# std_weight = torch.zeros((32, 32), dtype=torch.float32)
# std_weight[-4:-1, -4:-1] = 1.0

# summary.summarize(model_name, model, dataset_name, trainset, testset, poisoned_transform_train_index, poisoned_transform_test_index, std_pattern, std_weight)


# ========== ResNet-18_GTSRB ==========
model_name='ResNet-18'

model=core.models.ResNet(18, 43)
model.load_state_dict(torch.load('/disk/yamengxi/Backdoor/XAI/Backdoor_XAI/experiments/ResNet-18_GTSRB_BadNets_2022-05-25_02:55:53/ckpt_epoch_30.pth'), strict=False)

dataset_name='GTSRB'

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

poisoned_transform_train_index=2

poisoned_transform_test_index=2

std_pattern = torch.zeros((3, 32, 32), dtype=torch.float32)
std_pattern[:, -4:-1, -4:-1] = 255
std_weight = torch.zeros((32, 32), dtype=torch.float32)
std_weight[-4:-1, -4:-1] = 1.0

summary.summarize(model_name, model, dataset_name, trainset, testset, poisoned_transform_train_index, poisoned_transform_test_index, std_pattern, std_weight)