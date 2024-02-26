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
from TABOR.tabor import tabor


parser = argparse.ArgumentParser(description='TABOR Experiments Launcher for Trojaned Models on Datasets')
parser.add_argument('--model_type', default='core', type=str)
parser.add_argument('--model_name', default='ResNet-18', type=str)
parser.add_argument('--model_path', default='./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth', type=str)

parser.add_argument('--dataset_name', default='CIFAR-10', type=str)
parser.add_argument('--dataset_root_path', default='../datasets', type=str)

parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--seed', default=666, type=int)
parser.add_argument('--deterministic', action='store_true', default=False)

parser.add_argument('--y_target', default=0, type=int)
parser.add_argument('--trigger_path', default="./tests/square_trigger_3x3.png", type=str)

parser.add_argument('--TABOR_epochs', default=200, type=int)
parser.add_argument('--TABOR_lr', default=0.1, type=float)
parser.add_argument('--TABOR_optim', default='Adam', type=str)
parser.add_argument('--TABOR_schedule', default='', type=str)
parser.add_argument('--TABOR_gamma', default=0.1, type=float)
parser.add_argument('--TABOR_hyperparameters', default='1e-6,1e-5,1e-6', type=str)
parser.add_argument('--save_trigger_path', default='./TABOR/now_my_TABOR_experiments', type=str)

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
TABOR_epochs=args.TABOR_epochs
TABOR_lr=args.TABOR_lr
TABOR_optim=args.TABOR_optim
TABOR_schedule=args.TABOR_schedule
TABOR_schedule=[int(item) for item in TABOR_schedule.split(',')]
TABOR_gamma=args.TABOR_gamma
TABOR_hyperparameters=args.TABOR_hyperparameters
TABOR_hyperparameters=[float(item) for item in TABOR_hyperparameters.split(',')]
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


# Read trigger_img, get std_weight and std_pattern
trigger_img = cv2.imread(trigger_path) # (H, W) or (H, W, C), np.uint8, [0, 255]
if trigger_img.ndim == 3:
    if trigger_img.var(axis=2).sum() > 0:
        raise NotImplementedError(f"Unsupported color trigger_img, please use grayscale trigger_img.")
    else:
        trigger_img = trigger_img[:, :, 0] # (H, W, C) -> (H, W), np.uint8, [0, 255]
std_weight = torch.from_numpy(trigger_img).to(dtype=torch.float32) / 255.0 # (H, W), torch.float32, [0.0, 1.0]
trigger_pixel_num = std_weight.sum().item()
std_pattern = std_weight.unsqueeze(0) # (1, H, W), torch.float32, [0.0, 1.0]
std_pattern = std_pattern.repeat(data_shape[0], 1, 1) # (C, H, W), torch.float32, [0.0, 1.0]
std_pattern = normalize(std_pattern) # (C, H, W), torch.float32, [-x, y]
torch.manual_seed(seed)

if save_trigger_path.find("adv") >=0:
    save_trigger_path=osp.join(save_trigger_path, model_path.split('/')[-3], f"{model_path.split('/')[-2]}_epochs{TABOR_epochs}_lr{TABOR_lr}_{TABOR_optim}_schedule{args.TABOR_schedule}_gamma{TABOR_gamma}_hyperparameters{TABOR_hyperparameters}_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}")
else:
    save_trigger_path=osp.join(save_trigger_path, f"{model_path.split('/')[-2]}_epochs{TABOR_epochs}_lr{TABOR_lr}_{TABOR_optim}_schedule{args.TABOR_schedule}_gamma{TABOR_gamma}_hyperparameters{TABOR_hyperparameters}_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}")

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


blended = core.Blended(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=torch.nn.CrossEntropyLoss(),
    y_target=y_target,
    poisoned_rate=0.05,
    pattern=std_pattern,
    weight=std_weight,
    poisoned_transform_train_index=len(trainset.transform.transforms),
    poisoned_transform_test_index=len(testset.transform.transforms),
    poisoned_target_transform_index=0,
    schedule=None,
    seed=int(time.time()),
    deterministic=deterministic
)

_, poisoned_testset = blended.get_poisoned_dataset()

schedule = {
    'device': 'GPU',
    # 'CUDA_SELECTED_DEVICES': '0',

    'batch_size': batch_size,
    'num_workers': num_workers,

    'metric': 'BA',

    'save_dir': save_trigger_path,
    'experiment_name': f'_std_trigger_test_BA'
}
top1_correct, top5_correct, total_num, mean_ce_loss = core.utils.test(model, testset, schedule)

schedule = {
    'device': 'GPU',
    # 'CUDA_SELECTED_DEVICES': '0',

    'batch_size': batch_size,
    'num_workers': num_workers,

    'metric': 'ASR',

    'save_dir': save_trigger_path,
    'experiment_name': f'_std_trigger_test_ASR'
}
top1_correct, top5_correct, total_num, mean_ce_loss = core.utils.test(model, poisoned_testset, schedule)


distance_functions = [
    ["1-norm loss, reduction='mean'", norm_loss],
    ["1-norm distance, reduction='mean'", norm],
    ["cross entropy loss distance, reduction='mean'", binary_cross_entropy],
    ["lovasz loss distance, reduction='mean'", lovasz_hinge],
    ["chamfer distance, reduction='mean'", chamfer_distance]
]
mean_ce_losses = [mean_ce_loss]
mean_total_losses = [0.0]
distances = [[0.0] for item in distance_functions]
distances[0][0] = norm_loss(std_weight.clone().detach().cuda().unsqueeze(0), std_weight.clone().detach().cuda().unsqueeze(0)).cpu().item()


dataloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=False,
    pin_memory=True
)

def my_handler(signum, frame):
    global stop
    stop = True
signal.signal(signal.SIGINT, my_handler)

iteration=0
stop = False
while not stop:
    iteration += 1
    now_save_trigger_path=osp.join(save_trigger_path, f"{iteration}_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}")

    model = model.cuda()

    # mask (H, W), pattern (C, H, W)
    mask, pattern, mask_, mean_total_loss, mean_ce_loss, mean_regularization_loss = tabor(
        model=model,
        dataloader=dataloader,
        y_target=y_target,
        num_epochs=TABOR_epochs,
        lr=TABOR_lr,
        optim=TABOR_optim,
        schedule=TABOR_schedule,
        gamma=TABOR_gamma,
        hyperparameters=TABOR_hyperparameters,
        num_classes=num_classes,
        data_shape=data_shape,
        reversed_trigger_dir=now_save_trigger_path
    )

    blended = core.Blended(
        train_dataset=trainset,
        test_dataset=testset,
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        y_target=y_target,
        poisoned_rate=0.05,
        pattern=pattern.cpu(),
        weight=mask.cpu(),
        poisoned_transform_train_index=len(trainset.transform.transforms),
        poisoned_transform_test_index=len(testset.transform.transforms),
        poisoned_target_transform_index=0,
        schedule=None,
        seed=int(time.time()),
        deterministic=deterministic
    )

    _, poisoned_testset = blended.get_poisoned_dataset()

    schedule = {
        'device': 'GPU',

        'batch_size': batch_size,
        'num_workers': num_workers,

        'metric': 'ASR',

        'save_dir': now_save_trigger_path,
        'experiment_name': f'TABOR_trigger_test_ASR'
    }
    top1_correct, top5_correct, total_num, mean_ce_loss = core.utils.test(model, poisoned_testset, schedule)

    mean_ce_losses.append(mean_ce_loss)
    mean_total_losses.append(mean_ce_loss + mean_regularization_loss)

    for i in range(len(distances)):
        std_weight_ = std_weight.clone().detach().cuda().unsqueeze(0)
        weight_ = mask.clone().detach().cuda().unsqueeze(0)
        distances[i].append(distance_functions[i][1](weight_, std_weight_).cpu().item())


import matplotlib.pyplot as plt
distances = np.array(distances)
mean_ce_losses = np.array(mean_ce_losses)
mean_total_losses = np.array(mean_total_losses)
os.makedirs(osp.join(save_trigger_path, f'__summary_experiments_results'), exist_ok=True)
np.savez(osp.join(save_trigger_path, f'__summary_experiments_results', f"{dataset_name}_{iteration}.npz"), distances=distances, mean_ce_losses=mean_ce_losses)

for i in range(len(distances)):
    plt.figure(figsize=(16,9), dpi=600)
    plt.scatter(distances[i], mean_ce_losses, s=0.25)
    plt.savefig(osp.join(save_trigger_path, f'__summary_experiments_results', f"{dataset_name}_{iteration}_{distance_functions[i][0]}.png"))
    plt.close()

with open(osp.join(save_trigger_path, f'__summary_experiments_results', 'trigger_statistics.log'),'w') as f:
    f.write(f'lr:{TABOR_lr:.10f}\n')
    f.write(f'epochs:{TABOR_epochs}\n')
    f.write(f'schedule:{args.TABOR_schedule}\n')
    f.write(f'hyperparameters:{TABOR_hyperparameters}\n')

    f.write(f'TABOR样本数:{len(mean_ce_losses) - 1}\n')

    f.write(f'total Loss (min):{mean_total_losses[1:].min()}\n')
    f.write(f'total Loss (mean):{mean_total_losses[1:].mean()}\n')
    f.write(f'total Loss (max):{mean_total_losses[1:].max()}\n')
    f.write(f'total Loss (std):{mean_total_losses[1:].std()}\n')

    # f.write(f'Poisoned Loss:{mean_ce_losses[0]}\n')
    f.write(f'TABOR Loss (min):{mean_ce_losses[1:].min()}\n')
    f.write(f'TABOR Loss (mean):{mean_ce_losses[1:].mean()}\n')
    f.write(f'TABOR Loss (max):{mean_ce_losses[1:].max()}\n')
    f.write(f'TABOR Loss (std):{mean_ce_losses[1:].std()}\n')

    norm_losses = distances[0]
    f.write(f'TABOR norm loss (min):{norm_losses[1:].min()}\n')
    f.write(f'TABOR norm loss (mean):{norm_losses[1:].mean()}\n')
    f.write(f'TABOR norm loss (max):{norm_losses[1:].max()}\n')
    f.write(f'TABOR norm loss (std):{norm_losses[1:].std()}\n')

    f.write(f'{TABOR_lr:.10f}\n')
    f.write(f'{TABOR_epochs}\n')
    f.write(f'{args.TABOR_schedule}\n')
    f.write(f'{TABOR_hyperparameters}\n')
    f.write(f'{len(mean_ce_losses) - 1}\n')
    f.write(f'{mean_total_losses[1:].min()}\n')
    f.write(f'{mean_total_losses[1:].mean()}\n')
    f.write(f'{mean_total_losses[1:].max()}\n')
    f.write(f'{mean_total_losses[1:].std()}\n')
    f.write(f'{mean_ce_losses[1:].min()}\n')
    f.write(f'{mean_ce_losses[1:].mean()}\n')
    f.write(f'{mean_ce_losses[1:].max()}\n')
    f.write(f'{mean_ce_losses[1:].std()}\n')
    f.write(f'{norm_losses[1:].min()}\n')
    f.write(f'{norm_losses[1:].mean()}\n')
    f.write(f'{norm_losses[1:].max()}\n')
    f.write(f'{norm_losses[1:].std()}\n')


    # find top K
    norm_distances = distances[1]

    K = max(20, int(len(mean_ce_losses) * 0.1))
    K = min(K, len(mean_ce_losses))

    mean_ce_losses = torch.from_numpy(mean_ce_losses)
    topk = torch.topk(mean_ce_losses, K, dim=0, largest=True, sorted=True)
    f.write(f'\n=========={K} Largest loss TABOR trigger==========\n')
    for i in range(len(topk.indices)):
        f.write(f'trigger id:{topk.indices[i]}, mean_ce_loss:{mean_ce_losses[topk.indices[i]]}, distance:{norm_distances[topk.indices[i]]}\n')
    topk = torch.topk(mean_ce_losses, K, dim=0, largest=False, sorted=True)
    f.write(f'\n=========={K} Smallest loss TABOR trigger==========\n')
    for i in range(len(topk.indices)):
        f.write(f'trigger id:{topk.indices[i]}, mean_ce_loss:{mean_ce_losses[topk.indices[i]]}, distance:{norm_distances[topk.indices[i]]}\n')

    norm_distances = torch.from_numpy(norm_distances)
    topk = torch.topk(norm_distances, K, dim=0, largest=True, sorted=True)
    f.write(f'\n=========={K} Largest norm distance TABOR trigger==========\n')
    for i in range(len(topk.indices)):
        f.write(f'trigger id:{topk.indices[i]}, mean_ce_loss:{mean_ce_losses[topk.indices[i]]}, distance:{norm_distances[topk.indices[i]]}\n')
    topk = torch.topk(norm_distances, K, dim=0, largest=False, sorted=True)
    f.write(f'\n=========={K} Smallest norm distance TABOR trigger==========\n')
    for i in range(len(topk.indices)):
        f.write(f'trigger id:{topk.indices[i]}, mean_ce_loss:{mean_ce_losses[topk.indices[i]]}, distance:{norm_distances[topk.indices[i]]}\n')
